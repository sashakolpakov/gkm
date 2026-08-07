"""Append-only exposure accounting for Bongard drill and benchmark tasks.

An :class:`ExposureLedger` is a frozen value.  Recording access returns a new
ledger whose events form a SHA-256 chain; existing JSON files are never
overwritten.  This makes "unseen" and "sealed" concrete information-flow
claims rather than comments in a run script.
"""

from __future__ import annotations

from bongard.runtime_source_snapshot import capture_loaded_source


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence

from .corpus import _canonical_json_bytes

if TYPE_CHECKING:
    from .historical_exposure import HistoricalExposureSeed


LEDGER_SCHEMA = "gkm.bongard-exposure-ledger.v1"
PARTITION_SCHEMA = "gkm.bongard-task-partition.v1"
SEMANTIC_RESOLVER_POLICY_SCHEMA = "gkm.bongard-semantic-exposure-resolver.v2"

_CONTENT_ADDRESS_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_SEMANTIC_RESOLVER_POLICY = {
    "schema": SEMANTIC_RESOLVER_POLICY_SCHEMA,
    "parser": "bongard.cohorts.parse_official_task_id",
    "basic_exact": "one-key-per-named-shape-family",
    "basic_morphology_cluster": (
        "strip one terminal decimal variant suffix, including _newN; "
        "emit both exact and cluster keys"
    ),
    "basic_cluster_boundary": (
        "block a cluster if it contains a historically exposed family or "
        "crosses frozen drill/dev/sealed partitions"
    ),
    "abstract_pair": (
        "one-key-per-exact-ordered-pair; this certifies only an unseen ordered "
        "combination, not unseen component attributes"
    ),
    "abstract_singleton": "one-key-per-single-attribute",
    "freeform": "one-key-per-exact-generator-family",
}

_BASIC_TERMINAL_VARIANT_RE = re.compile(r"(?:_new)?[0-9]+\Z")


class ExposureError(RuntimeError):
    """Base class for malformed ledgers and access-policy failures."""


class ExposureIntegrityError(ExposureError):
    """A serialized ledger is malformed or its digest chain is broken."""


class ExposureViolation(ExposureError):
    """A requested access would break an unseen or sealed-set promise."""


def _address(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _clean_ids(values: Iterable[str], *, label: str) -> tuple[str, ...]:
    cleaned = tuple(values)
    if not all(isinstance(value, str) and value for value in cleaned):
        raise ExposureIntegrityError(f"{label} must contain non-empty strings")
    if len(cleaned) != len(set(cleaned)):
        raise ExposureIntegrityError(f"{label} contains duplicate identifiers")
    return tuple(sorted(cleaned))


def _id_sequence(value: Any, *, label: str) -> tuple[str, ...]:
    """Normalise one legacy scalar id or an iterable of ids."""

    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    try:
        return tuple(value)
    except TypeError as exc:
        raise ExposureIntegrityError(f"{label} must be a string or iterable of strings") from exc


def task_id_from_panel_id(panel_id: str) -> str:
    """Return the task component of ``family/task/label/file.png``.

    Unknown panel-id formats are rejected instead of weakening task-level
    sealing when the ledger cannot prove which task owns a panel.
    """

    pieces = panel_id.split("/")
    if (
        len(pieces) != 4
        or pieces[0] not in {"ff", "bd", "hd"}
        or pieces[2] not in {"0", "1"}
    ):
        raise ExposureIntegrityError(f"non-canonical panel id: {panel_id!r}")
    if (
        not pieces[1].startswith(f"{pieces[0]}_")
        or re.fullmatch(r"[0-6]\.png", pieces[3]) is None
    ):
        raise ExposureIntegrityError(f"non-canonical panel id: {panel_id!r}")
    return pieces[1]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True, order=True)
class SemanticDisclosureKey:
    """One collision domain opened by viewing an official task."""

    kind: str
    concepts: tuple[str, ...]

    def __post_init__(self) -> None:
        arities = {
            "basic_family": 1,
            "basic_morphology_cluster": 1,
            "abstract_pair": 2,
            "abstract_attribute": 1,
            "freeform_family": 1,
        }
        if self.kind not in arities or len(self.concepts) != arities[self.kind]:
            raise ExposureIntegrityError("invalid semantic disclosure key")
        if any(not isinstance(concept, str) or not concept for concept in self.concepts):
            raise ExposureIntegrityError(
                "semantic disclosure concepts must be non-empty strings"
            )

    def to_dict(self) -> dict[str, object]:
        return {"kind": self.kind, "concepts": list(self.concepts)}


def basic_morphology_cluster_id(shape_family: str) -> str:
    """Return the conservative v2 lexical cluster for one Basic family.

    ShapeBongard's Basic inventory contains numbered variants such as
    ``advanced_lamp1`` through ``advanced_lamp7`` and ``bird1`` through
    ``bird8``.  Treating those names as independent semantic concepts leaked
    morphology across historical/drill/dev/sealed boundaries.  V2 removes one
    terminal decimal variant suffix (and the upstream ``_newN`` spelling).

    This is intentionally conservative and metadata-only.  It never reads a
    panel or an action program, and over-grouping only reduces the eligible
    reservoir.
    """

    if not isinstance(shape_family, str) or not shape_family:
        raise ExposureIntegrityError("Basic shape family must be a non-empty string")
    cluster = _BASIC_TERMINAL_VARIANT_RE.sub("", shape_family)
    if not cluster:
        raise ExposureIntegrityError(
            f"Basic shape family has no morphology stem: {shape_family!r}"
        )
    return cluster


def _basic_morphology_clusters(
    seed: "HistoricalExposureSeed",
) -> dict[str, tuple[str, ...]]:
    vocabulary = set(seed.basic_shape_families) | set(
        seed.unused_basic_shape_families
    )
    clusters: dict[str, list[str]] = {}
    for family in sorted(vocabulary):
        clusters.setdefault(basic_morphology_cluster_id(family), []).append(family)
    return {name: tuple(members) for name, members in sorted(clusters.items())}


def semantic_policy_blocked_keys(
    seed: "HistoricalExposureSeed",
) -> tuple[SemanticDisclosureKey, ...]:
    """Return v2 Basic clusters that cannot support an independence claim.

    A cluster is blocked when any member was historically exposed, or when its
    otherwise-unused members were split across more than one frozen cohort.
    The latter prevents a numbered morphology series from leaking between
    drill, development, and sealed evaluation merely because its suffixes
    received different hash ranks.
    """

    exposed = set(seed.basic_shape_families)
    partition_of = {
        family: cohort
        for cohort, families in (
            ("drill", seed.partition.drill),
            ("dev", seed.partition.dev),
            ("sealed", seed.partition.sealed),
        )
        for family in families
    }
    blocked: list[SemanticDisclosureKey] = []
    for cluster, members in _basic_morphology_clusters(seed).items():
        unused_partitions = {
            partition_of[family] for family in members if family in partition_of
        }
        if any(family in exposed for family in members) or len(unused_partitions) > 1:
            blocked.append(
                SemanticDisclosureKey("basic_morphology_cluster", (cluster,))
            )
    return tuple(blocked)


@dataclass(frozen=True)
class SemanticExposureResolution:
    """Derived semantic keys plus the exact interpretation that produced them."""

    task_ids: tuple[str, ...]
    semantic_keys: tuple[SemanticDisclosureKey, ...]
    historical_seed_digest: str
    resolver_policy_digest: str
    ledger_digest: str

    def to_dict(self) -> dict[str, object]:
        return {
            "task_ids": list(self.task_ids),
            "semantic_keys": [key.to_dict() for key in self.semantic_keys],
            "historical_seed_digest": self.historical_seed_digest,
            "resolver_policy_digest": self.resolver_policy_digest,
            "ledger_digest": self.ledger_digest,
        }


def _semantic_seed_projection(seed: "HistoricalExposureSeed") -> dict[str, object]:
    """Return exactly the frozen parser vocabulary relevant to live exposure."""

    clusters = _basic_morphology_clusters(seed)
    return {
        "historical_seed_digest": seed.seed_digest,
        "basic_shape_families": sorted(
            set(seed.basic_shape_families) | set(seed.unused_basic_shape_families)
        ),
        "basic_morphology_clusters": [
            {"cluster": cluster, "members": list(members)}
            for cluster, members in clusters.items()
        ],
        "blocked_basic_morphology_clusters": [
            key.concepts[0] for key in semantic_policy_blocked_keys(seed)
        ],
        "abstract_attributes": sorted(set(seed.abstract_attributes)),
        "admissible_abstract_pairs": [
            list(pair) for pair in sorted(set(seed.admissible_abstract_pairs))
        ],
    }


def semantic_resolver_policy_digest(seed: "HistoricalExposureSeed") -> str:
    """Bind the parser policy and the vocabulary used to interpret task IDs."""

    # Imports are local so loading exposure ledgers never imports the historical
    # audit machinery or triggers its filesystem-backed default seed loader.
    from .historical_exposure import HistoricalExposureSeed

    if not isinstance(seed, HistoricalExposureSeed):
        raise ExposureIntegrityError(
            "semantic resolution requires a validated HistoricalExposureSeed"
        )
    if _CONTENT_ADDRESS_RE.fullmatch(seed.seed_digest) is None:
        raise ExposureIntegrityError(
            "historical semantic seed digest is not a canonical content address"
        )
    return _address(
        {
            **_SEMANTIC_RESOLVER_POLICY,
            "vocabulary": _semantic_seed_projection(seed),
        }
    )


def _semantic_binding(
    seed: "HistoricalExposureSeed",
    *,
    expected_historical_seed_digest: str | None,
    expected_resolver_policy_digest: str | None,
) -> tuple[str, str]:
    if (
        expected_historical_seed_digest is None
        and expected_resolver_policy_digest is None
    ):
        raise ExposureViolation(
            "semantic resolution requires a precommitted historical seed or "
            "resolver-policy digest"
        )
    policy_digest = semantic_resolver_policy_digest(seed)
    if (
        expected_historical_seed_digest is not None
        and expected_historical_seed_digest != seed.seed_digest
    ):
        raise ExposureViolation(
            "historical semantic seed differs from the precommitted digest"
        )
    if (
        expected_resolver_policy_digest is not None
        and expected_resolver_policy_digest != policy_digest
    ):
        raise ExposureViolation(
            "semantic resolver policy differs from the precommitted digest"
        )
    return seed.seed_digest, policy_digest


def _semantic_keys_for_task_id(
    task_id: str,
    seed: "HistoricalExposureSeed",
    *,
    requested: bool,
) -> tuple[SemanticDisclosureKey, ...]:
    # Importing here avoids an exposure -> cohorts -> historical_exposure
    # import dependency during ordinary ledger loading and exact-ID checks.
    from .cohorts import CohortError, parse_official_task_id

    try:
        parsed = parse_official_task_id(task_id, seed)
    except (CohortError, TypeError, ValueError) as exc:
        error = ExposureViolation if requested else ExposureIntegrityError
        role = "requested" if requested else "recorded"
        raise error(
            f"cannot derive {role} task semantics from {task_id!r}: {exc}"
        ) from exc

    if parsed.family == "bd":
        return tuple(
            sorted(
                {
                    key
                    for concept in parsed.concepts
                    for key in (
                        SemanticDisclosureKey("basic_family", (concept,)),
                        SemanticDisclosureKey(
                            "basic_morphology_cluster",
                            (basic_morphology_cluster_id(concept),),
                        ),
                    )
                }
            )
        )
    if parsed.family == "hd":
        kind = "abstract_pair" if len(parsed.concepts) == 2 else "abstract_attribute"
        return (SemanticDisclosureKey(kind, parsed.concepts),)
    if parsed.family == "ff":
        return (SemanticDisclosureKey("freeform_family", parsed.concepts),)
    raise ExposureIntegrityError(
        f"frozen task parser returned unsupported family {parsed.family!r}"
    )


@dataclass(frozen=True)
class ExposureEvent:
    sequence: int
    observed_at: str
    phase: str
    actor: str
    purpose: str
    task_ids: tuple[str, ...]
    panel_ids: tuple[str, ...]
    source: str | None
    previous_digest: str | None
    digest: str

    def content_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "observed_at": self.observed_at,
            "phase": self.phase,
            "actor": self.actor,
            "purpose": self.purpose,
            "task_ids": list(self.task_ids),
            "panel_ids": list(self.panel_ids),
            "source": self.source,
            "previous_digest": self.previous_digest,
        }

    def to_dict(self) -> dict[str, Any]:
        result = self.content_dict()
        result["digest"] = self.digest
        return result

    @classmethod
    def create(
        cls,
        *,
        sequence: int,
        observed_at: str,
        phase: str,
        actor: str,
        purpose: str,
        task_ids: Iterable[str] = (),
        panel_ids: Iterable[str] = (),
        source: str | None = None,
        previous_digest: str | None,
    ) -> "ExposureEvent":
        tasks = _clean_ids(task_ids, label="task_ids")
        panels = _clean_ids(panel_ids, label="panel_ids")
        for panel_id in panels:
            task_id_from_panel_id(panel_id)
        if not tasks and not panels:
            raise ExposureIntegrityError("an exposure event must name a task or panel")
        for name, value in (("phase", phase), ("actor", actor), ("purpose", purpose)):
            if not isinstance(value, str) or not value.strip():
                raise ExposureIntegrityError(f"{name} must be a non-empty string")
        if not isinstance(observed_at, str) or not observed_at:
            raise ExposureIntegrityError("observed_at must be a non-empty ISO timestamp")
        if source is not None and (not isinstance(source, str) or not source.strip()):
            raise ExposureIntegrityError("source must be null or a non-empty string")
        content = {
            "sequence": sequence,
            "observed_at": observed_at,
            "phase": phase,
            "actor": actor,
            "purpose": purpose,
            "task_ids": list(tasks),
            "panel_ids": list(panels),
            "source": source,
            "previous_digest": previous_digest,
        }
        return cls(
            sequence=sequence,
            observed_at=observed_at,
            phase=phase,
            actor=actor,
            purpose=purpose,
            task_ids=tasks,
            panel_ids=panels,
            source=source,
            previous_digest=previous_digest,
            digest=_address(content),
        )

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ExposureEvent":
        required = {
            "sequence",
            "observed_at",
            "phase",
            "actor",
            "purpose",
            "task_ids",
            "panel_ids",
            "source",
            "previous_digest",
            "digest",
        }
        if set(raw) != required:
            raise ExposureIntegrityError(
                f"event fields differ from schema: missing={required - set(raw)}, extra={set(raw) - required}"
            )
        if not isinstance(raw["sequence"], int) or raw["sequence"] < 0:
            raise ExposureIntegrityError("event sequence must be a non-negative integer")
        if not isinstance(raw["task_ids"], list) or not isinstance(raw["panel_ids"], list):
            raise ExposureIntegrityError("serialized task_ids and panel_ids must be lists")
        event = cls.create(
            sequence=raw["sequence"],
            observed_at=raw["observed_at"],
            phase=raw["phase"],
            actor=raw["actor"],
            purpose=raw["purpose"],
            task_ids=raw["task_ids"],
            panel_ids=raw["panel_ids"],
            source=raw["source"],
            previous_digest=raw["previous_digest"],
        )
        if raw["digest"] != event.digest:
            raise ExposureIntegrityError(
                f"event {event.sequence} digest mismatch: {raw['digest']!r} != {event.digest!r}"
            )
        return event


@dataclass(frozen=True)
class ExposureLedger:
    """Persistent value recording every authorized visual disclosure."""

    corpus_digest: str
    events: tuple[ExposureEvent, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.corpus_digest, str) or not self.corpus_digest.startswith("sha256:"):
            raise ExposureIntegrityError("corpus_digest must be a sha256: content address")
        previous: str | None = None
        for sequence, event in enumerate(self.events):
            if event.sequence != sequence:
                raise ExposureIntegrityError(
                    f"event sequence is {event.sequence}, expected {sequence}"
                )
            if event.previous_digest != previous:
                raise ExposureIntegrityError(
                    f"event {sequence} points to {event.previous_digest!r}, expected {previous!r}"
                )
            if _address(event.content_dict()) != event.digest:
                raise ExposureIntegrityError(f"event {sequence} digest is invalid")
            previous = event.digest

    @classmethod
    def create(cls, corpus_digest: str) -> "ExposureLedger":
        return cls(corpus_digest=corpus_digest)

    @property
    def digest(self) -> str:
        return _address(self.content_dict())

    @property
    def exposed_task_ids(self) -> frozenset[str]:
        result: set[str] = set()
        for event in self.events:
            result.update(event.task_ids)
            result.update(task_id_from_panel_id(panel_id) for panel_id in event.panel_ids)
        return frozenset(result)

    @property
    def explicitly_exposed_panel_ids(self) -> frozenset[str]:
        return frozenset(panel_id for event in self.events for panel_id in event.panel_ids)

    def derive_exposed_semantic_keys(
        self,
        *,
        historical_seed: "HistoricalExposureSeed",
        expected_historical_seed_digest: str | None = None,
        expected_resolver_policy_digest: str | None = None,
    ) -> SemanticExposureResolution:
        """Derive live semantic exposure without changing ledger schema v1.

        Panel-only events first imply their owning exact task through
        :attr:`exposed_task_ids`.  Every resulting task ID must parse under the
        bound frozen vocabulary; an old malformed ID makes the semantic view
        unavailable rather than being silently ignored.
        """

        seed_digest, policy_digest = _semantic_binding(
            historical_seed,
            expected_historical_seed_digest=expected_historical_seed_digest,
            expected_resolver_policy_digest=expected_resolver_policy_digest,
        )
        tasks = tuple(sorted(self.exposed_task_ids))
        keys = tuple(
            sorted(
                {
                    key
                    for task_id in tasks
                    for key in _semantic_keys_for_task_id(
                        task_id,
                        historical_seed,
                        requested=False,
                    )
                }
            )
        )
        return SemanticExposureResolution(
            task_ids=tasks,
            semantic_keys=keys,
            historical_seed_digest=seed_digest,
            resolver_policy_digest=policy_digest,
            ledger_digest=self.digest,
        )

    def assert_semantically_unseen(
        self,
        *,
        task_ids: Iterable[str],
        historical_seed: "HistoricalExposureSeed",
        expected_historical_seed_digest: str | None = None,
        expected_resolver_policy_digest: str | None = None,
    ) -> SemanticExposureResolution:
        """Reject requested tasks sharing a live semantic disclosure key.

        The returned resolution is a small receipt callers can bind into a
        later run schema.  This primitive deliberately does not record the
        access; the CLI can perform the check before its existing ``record``
        transition once that integration is versioned.
        """

        exposed = self.derive_exposed_semantic_keys(
            historical_seed=historical_seed,
            expected_historical_seed_digest=expected_historical_seed_digest,
            expected_resolver_policy_digest=expected_resolver_policy_digest,
        )
        requested_tasks = _clean_ids(task_ids, label="task_ids")
        requested_keys = tuple(
            sorted(
                {
                    key
                    for task_id in requested_tasks
                    for key in _semantic_keys_for_task_id(
                        task_id,
                        historical_seed,
                        requested=True,
                    )
                }
            )
        )
        collisions = set(requested_keys) & (
            set(exposed.semantic_keys) | set(semantic_policy_blocked_keys(historical_seed))
        )
        if collisions:
            raise ExposureViolation(
                "requested task semantics are not unseen: "
                f"keys={[key.to_dict() for key in sorted(collisions)]}"
            )
        return SemanticExposureResolution(
            task_ids=requested_tasks,
            semantic_keys=requested_keys,
            historical_seed_digest=exposed.historical_seed_digest,
            resolver_policy_digest=exposed.resolver_policy_digest,
            ledger_digest=self.digest,
        )

    def content_dict(self) -> dict[str, Any]:
        return {
            "schema": LEDGER_SCHEMA,
            "corpus_digest": self.corpus_digest,
            "events": [event.to_dict() for event in self.events],
        }

    def to_dict(self) -> dict[str, Any]:
        result = self.content_dict()
        result["ledger_digest"] = self.digest
        return result

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, indent=2, ensure_ascii=False) + "\n"

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ExposureLedger":
        required = {"schema", "corpus_digest", "events", "ledger_digest"}
        if set(raw) != required:
            raise ExposureIntegrityError(
                f"ledger fields differ from schema: missing={required - set(raw)}, extra={set(raw) - required}"
            )
        if raw["schema"] != LEDGER_SCHEMA:
            raise ExposureIntegrityError(f"unsupported ledger schema: {raw['schema']!r}")
        if not isinstance(raw["events"], list):
            raise ExposureIntegrityError("ledger events must be a list")
        events = tuple(ExposureEvent.from_dict(event) for event in raw["events"])
        ledger = cls(corpus_digest=raw["corpus_digest"], events=events)
        if raw["ledger_digest"] != ledger.digest:
            raise ExposureIntegrityError(
                f"ledger digest mismatch: {raw['ledger_digest']!r} != {ledger.digest!r}"
            )
        return ledger

    @classmethod
    def load(cls, path: str | Path) -> "ExposureLedger":
        source = Path(path)
        try:
            raw = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ExposureIntegrityError(f"cannot read exposure ledger {source}: {exc}") from exc
        if not isinstance(raw, dict):
            raise ExposureIntegrityError("exposure ledger JSON must be an object")
        return cls.from_dict(raw)

    def write_once(self, path: str | Path) -> Path:
        """Create *path* atomically; never replace an existing ledger file."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_json()
        try:
            with destination.open("x", encoding="utf-8") as handle:
                handle.write(payload)
        except FileExistsError:
            # Content-addressed persistence is idempotent, but a collision with
            # different bytes is an integrity error rather than an overwrite.
            if destination.read_text(encoding="utf-8") != payload:
                raise ExposureIntegrityError(
                    f"refusing to overwrite different ledger at {destination}"
                )
        return destination

    def write_content_addressed(self, directory: str | Path) -> Path:
        filename = self.digest.removeprefix("sha256:") + ".exposure.json"
        return self.write_once(Path(directory) / filename)

    def assert_corpus(self, corpus_digest: str) -> None:
        if corpus_digest != self.corpus_digest:
            raise ExposureViolation(
                f"ledger belongs to {self.corpus_digest}, not requested corpus {corpus_digest}"
            )

    def assert_unseen(
        self,
        *,
        task_ids: Iterable[str] = (),
        panel_ids: Iterable[str] = (),
    ) -> None:
        tasks = set(_clean_ids(task_ids, label="task_ids"))
        panels = set(_clean_ids(panel_ids, label="panel_ids"))
        panel_tasks = {task_id_from_panel_id(panel_id) for panel_id in panels}
        task_hits = (tasks | panel_tasks) & set(self.exposed_task_ids)
        panel_hits = panels & set(self.explicitly_exposed_panel_ids)
        if task_hits or panel_hits:
            raise ExposureViolation(
                "requested data are not unseen: "
                f"tasks={sorted(task_hits)}, panels={sorted(panel_hits)}"
            )

    @staticmethod
    def assert_not_sealed(
        *,
        task_ids: Iterable[str] = (),
        panel_ids: Iterable[str] = (),
        sealed_task_ids: Iterable[str] = (),
        sealed_panel_ids: Iterable[str] = (),
    ) -> None:
        tasks = set(_clean_ids(task_ids, label="task_ids"))
        panels = set(_clean_ids(panel_ids, label="panel_ids"))
        sealed_tasks = set(_clean_ids(sealed_task_ids, label="sealed_task_ids"))
        sealed_panels = set(_clean_ids(sealed_panel_ids, label="sealed_panel_ids"))
        requested_panel_tasks = {task_id_from_panel_id(panel_id) for panel_id in panels}
        sealed_panel_tasks = {task_id_from_panel_id(panel_id) for panel_id in sealed_panels}
        task_hits = (tasks | requested_panel_tasks) & sealed_tasks
        task_hits.update(tasks & sealed_panel_tasks)
        panel_hits = panels & sealed_panels
        if task_hits or panel_hits:
            raise ExposureViolation(
                "requested access intersects the sealed set: "
                f"tasks={sorted(task_hits)}, panels={sorted(panel_hits)}"
            )

    def record(
        self,
        *,
        phase: str,
        actor: str,
        purpose: str,
        task_ids: Iterable[str] = (),
        panel_ids: Iterable[str] = (),
        source: str | None = None,
        observed_at: str | None = None,
        known_task_ids: Iterable[str] | None = None,
        known_panel_ids: Iterable[str] | None = None,
        sealed_task_ids: Iterable[str] = (),
        sealed_panel_ids: Iterable[str] = (),
        allow_sealed: bool = False,
        require_unseen: bool = False,
    ) -> "ExposureLedger":
        """Validate an access request and return a ledger containing it.

        ``allow_sealed`` should only be set by the one-shot benchmark opener.
        Callers that claim fresh data set ``require_unseen=True``; a prior
        task-level disclosure also makes every panel of that task non-unseen.
        """

        tasks = _clean_ids(task_ids, label="task_ids")
        panels = _clean_ids(panel_ids, label="panel_ids")
        panel_tasks = {task_id_from_panel_id(panel_id) for panel_id in panels}
        if known_task_ids is not None:
            known_tasks = set(known_task_ids)
            unknown_tasks = (set(tasks) | panel_tasks) - known_tasks
            if unknown_tasks:
                raise ExposureViolation(f"unknown task ids: {sorted(unknown_tasks)}")
        if known_panel_ids is not None:
            unknown_panels = set(panels) - set(known_panel_ids)
            if unknown_panels:
                raise ExposureViolation(f"unknown panel ids: {sorted(unknown_panels)}")
        if not allow_sealed:
            self.assert_not_sealed(
                task_ids=tasks,
                panel_ids=panels,
                sealed_task_ids=sealed_task_ids,
                sealed_panel_ids=sealed_panel_ids,
            )
        if require_unseen:
            self.assert_unseen(task_ids=tasks, panel_ids=panels)
        event = ExposureEvent.create(
            sequence=len(self.events),
            observed_at=observed_at or _utc_now(),
            phase=phase,
            actor=actor,
            purpose=purpose,
            task_ids=tasks,
            panel_ids=panels,
            source=source,
            previous_digest=self.events[-1].digest if self.events else None,
        )
        return ExposureLedger(corpus_digest=self.corpus_digest, events=self.events + (event,))

    def unseen_task_ids(self, eligible_task_ids: Iterable[str]) -> tuple[str, ...]:
        return tuple(sorted(set(eligible_task_ids) - set(self.exposed_task_ids)))


@dataclass(frozen=True)
class TaskPartition:
    """Content-addressed deterministic drill/dev/sealed task partition."""

    namespace: str
    drill: tuple[str, ...]
    dev: tuple[str, ...]
    sealed: tuple[str, ...]
    digest: str

    def content_dict(self) -> dict[str, Any]:
        return {
            "schema": PARTITION_SCHEMA,
            "namespace": self.namespace,
            "drill": list(self.drill),
            "dev": list(self.dev),
            "sealed": list(self.sealed),
        }

    def to_dict(self) -> dict[str, Any]:
        result = self.content_dict()
        result["digest"] = self.digest
        return result


def deterministic_partition(
    eligible_task_ids: Iterable[str],
    *,
    drill_count: int,
    dev_count: int,
    sealed_count: int | None = None,
    namespace: str = "bongard-unused-v1",
) -> TaskPartition:
    """Partition eligible IDs by a stable SHA-256 rank, never input order."""

    eligible = tuple(eligible_task_ids)
    if len(eligible) != len(set(eligible)):
        raise ValueError("eligible_task_ids contains duplicates")
    if not namespace:
        raise ValueError("namespace must be non-empty")
    if not isinstance(drill_count, int) or not isinstance(dev_count, int):
        raise TypeError("partition counts must be integers")
    if drill_count < 0 or dev_count < 0:
        raise ValueError("partition counts cannot be negative")
    if sealed_count is None:
        sealed_count = len(eligible) - drill_count - dev_count
    if not isinstance(sealed_count, int):
        raise TypeError("sealed_count must be an integer or None")
    if sealed_count < 0:
        raise ValueError("sealed_count cannot be negative")
    if drill_count + dev_count + sealed_count != len(eligible):
        raise ValueError(
            "drill_count + dev_count + sealed_count must exhaust eligible_task_ids"
        )

    def rank(task_id: str) -> tuple[str, str]:
        if not isinstance(task_id, str) or not task_id:
            raise ValueError("eligible task ids must be non-empty strings")
        digest = hashlib.sha256(f"{namespace}/{task_id}".encode("utf-8")).hexdigest()
        return digest, task_id

    ordered = tuple(task_id for _, task_id in sorted(rank(task_id) for task_id in eligible))
    drill = ordered[:drill_count]
    dev = ordered[drill_count : drill_count + dev_count]
    sealed = ordered[drill_count + dev_count :]
    content = {
        "schema": PARTITION_SCHEMA,
        "namespace": namespace,
        "drill": list(drill),
        "dev": list(dev),
        "sealed": list(sealed),
    }
    return TaskPartition(
        namespace=namespace,
        drill=drill,
        dev=dev,
        sealed=sealed,
        digest=_address(content),
    )


def import_historical_exposures(
    ledger: ExposureLedger,
    records: Sequence[str] | Sequence[Mapping[str, Any]] | Mapping[str, Any],
    *,
    source: str,
    default_actor: str = "legacy-import",
    default_purpose: str = "known prior exposure",
    observed_at: str = "1970-01-01T00:00:00Z",
    known_task_ids: Iterable[str] | None = None,
    known_panel_ids: Iterable[str] | None = None,
) -> ExposureLedger:
    """Import known old disclosures without pretending their time is known.

    Accepted forms are a sequence of task-id strings, a single mapping with
    ``task_ids``/``panel_ids``, a mapping with an ``events`` list, or a sequence
    of event mappings.  Every imported event is marked ``phase=historical`` by
    default and records the caller-supplied provenance ``source``.
    """

    if not source:
        raise ValueError("historical exposure source must be non-empty")
    if isinstance(records, Mapping) and "events" in records:
        nested = records["events"]
        if not isinstance(nested, list):
            raise ExposureIntegrityError("historical events must be a list")
        normalised: Sequence[Any] = nested
    elif isinstance(records, Mapping):
        normalised = [records]
    else:
        normalised = records

    if not normalised:
        return ledger

    if all(isinstance(record, str) for record in normalised):
        return ledger.record(
            phase="historical",
            actor=default_actor,
            purpose=default_purpose,
            task_ids=normalised,  # type: ignore[arg-type]
            source=source,
            observed_at=observed_at,
            known_task_ids=known_task_ids,
            known_panel_ids=known_panel_ids,
        )

    result = ledger
    for raw in normalised:
        if not isinstance(raw, Mapping):
            raise ExposureIntegrityError("historical records must all be strings or mappings")
        task_ids = _id_sequence(raw.get("task_ids", ()), label="historical task_ids")
        panel_ids = _id_sequence(raw.get("panel_ids", ()), label="historical panel_ids")
        if "task_id" in raw:
            task_ids = task_ids + (raw["task_id"],)
        if "panel_id" in raw:
            panel_ids = panel_ids + (raw["panel_id"],)
        result = result.record(
            phase=str(raw.get("phase", "historical")),
            actor=str(raw.get("actor", default_actor)),
            purpose=str(raw.get("purpose", default_purpose)),
            task_ids=task_ids,
            panel_ids=panel_ids,
            source=str(raw.get("source", source)),
            observed_at=str(raw.get("observed_at", observed_at)),
            known_task_ids=known_task_ids,
            known_panel_ids=known_panel_ids,
        )
    return result

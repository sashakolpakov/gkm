"""Model-free precommit for one exploratory ShapeBongard smoke task.

Selection is metadata-only and deliberately reuses an exposed Basic generator
cluster.  The core durably persists the exact-ID exposure successor before any
selected panel is hashed.  Serialization redacts query-enumerable commitments
and retains four false scientific-authorization flags.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, fields
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from types import MappingProxyType
from typing import Any, Mapping

from bongard import benchmark
from bongard.artifacts import canonical_digest, canonical_json
from bongard.cohorts import build_cohort_report, parse_official_task_id
from bongard.corpus import (
    EXPECTED_FAMILY_COUNTS,
    EXPECTED_REGIME_COUNTS,
    EXPECTED_SPLIT_COUNTS,
    FAMILIES,
    BongardTask,
    CorpusManifest,
    ShapeBongardCorpus,
    TaskManifest,
)
from bongard.exposure import (
    ExposureLedger,
    basic_morphology_cluster_id,
    semantic_policy_blocked_keys,
    semantic_resolver_policy_digest,
)
from bongard.historical_exposure import load_historical_exposure
from bongard.semantic_calibration_campaign import semantic_generator_cluster_id


ATOMIC_SMOKE_SELECTION_SCHEMA = "gkm.bongard-atomic-smoke-selection.v2"
ATOMIC_SMOKE_PRECOMMIT_SCHEMA = "gkm.bongard-atomic-smoke-precommit.v2"
ATOMIC_SMOKE_DEVELOPMENT_MANIFEST_SCHEMA = (
    "gkm.bongard-atomic-smoke-development-manifest-public.v2"
)
ATOMIC_SMOKE_PERSISTENCE_SCHEMA = "gkm.bongard-atomic-smoke-persistence.v1"
ATOMIC_SMOKE_SELECTION_POLICY = (
    "repeated-generator-drill-exact-unseen-train-bd/v1"
)
ATOMIC_SMOKE_SAMPLE_SIZE = 1
OFFICIAL_A3_SUCCESSOR_UNIVERSE_COUNT = 10
OFFICIAL_A3_SUCCESSOR_UNIVERSE_DIGEST = (
    "sha256:3246017440379de1e49f695503536f75062626d2de36bdab9112e96281e269a8"
)
OFFICIAL_CORPUS_MANIFEST_DIGEST = (
    "sha256:6fa51548520190a412812ba8f872dc3c7a7a2b2c47c0e42a4d9f6df351dce138"
)
OFFICIAL_A3_SUCCESSOR_LEDGER_DIGEST = (
    "sha256:7c85922f238eb121a30d441ccf3528c665037a34240e07a06feef01cc30cd7c4"
)
OFFICIAL_SPLIT_SOURCE_DIGEST = (
    "sha256:ebb9cd474478e0776dff539951070db2c96b9b312c4b0b073689d20792ed7230"
)
OFFICIAL_HISTORICAL_SEED_DIGEST = (
    "sha256:0dfa94ada526e47cfe41745125609b7b4e669e1e003d2f5366f740ff50e02ebf"
)
OFFICIAL_RESOLVER_POLICY_DIGEST = (
    "sha256:48598ae580a2f88aee7652d36fd386d54a8e4265b040bf1313f558508f47af9a"
)
OFFICIAL_BLOCKED_MORPHOLOGY_POLICY_DIGEST = (
    "sha256:ab8e37b7c6281e6adbc6a24e779280fd979f475f815f7aa71a76c2e0bdc5f6b8"
)
OFFICIAL_RELEASE_DESCRIPTOR_DIGEST = (
    "sha256:4d5fb0ad6093ab32e8a8ac0ca5a3405482e1218994f9d257238e4a09fc56cd2b"
)
OFFICIAL_CORPUS_LAYOUT = "archive"
OFFICIAL_TASK_COUNT = 12_000
OFFICIAL_FAMILY_COUNTS = tuple(EXPECTED_FAMILY_COUNTS.items())
OFFICIAL_SPLIT_COUNTS = tuple(EXPECTED_SPLIT_COUNTS.items())
OFFICIAL_REGIME_COUNTS = tuple(EXPECTED_REGIME_COUNTS.items())
ATOMIC_SMOKE_EXPOSURE_PHASE = "atomic-smoke-precommit"
ATOMIC_SMOKE_EXPOSURE_PURPOSE = (
    "exploratory-repeated-generator-exact-task-smoke"
)

_ADDRESS = re.compile(r"sha256:[0-9a-f]{64}\Z")
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")


class AtomicSmokePrecommitError(ValueError):
    """Selection, persistence, decoding, or replay violated the protocol."""


def _address(value: object, label: str) -> str:
    if not isinstance(value, str) or _ADDRESS.fullmatch(value) is None:
        raise AtomicSmokePrecommitError(f"{label} must be a sha256: content address")
    return value


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise AtomicSmokePrecommitError(f"{label} must be a lowercase SHA-256")
    return value


def _text(value: object, label: str, *, limit: int = 4096) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or value != value.strip()
        or len(value.encode("utf-8")) > limit
        or "\x00" in value
    ):
        raise AtomicSmokePrecommitError(f"{label} must be bounded exact text")
    return value


def _mapping(value: object, fields: frozenset[str], label: str) -> Mapping[str, Any]:
    if (
        not isinstance(value, Mapping)
        or any(not isinstance(key, str) for key in value)
        or set(value) != fields
    ):
        raise AtomicSmokePrecommitError(f"{label} fields differ from the static schema")
    return value


def _canonical_clone(value: object, label: str) -> Any:
    try:
        return json.loads(canonical_json(value))
    except (TypeError, ValueError, UnicodeError) as exc:
        raise AtomicSmokePrecommitError(f"{label} is not finite canonical JSON") from exc


def _freeze_json(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze_json(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _content_address(value: object) -> str:
    return "sha256:" + canonical_digest(value)


def _exact_int(value: object, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise AtomicSmokePrecommitError(
            f"{label} must be an exact integer >= {minimum}"
        )
    return value


def _seed_commitment(seed: str) -> str:
    _text(seed, "post-freeze selection seed")
    return canonical_digest({
        "schema": "gkm.bongard-atomic-smoke-seed-commitment.v1", "seed": seed,
    })


def _seed_rank(seed: str, task_id: str) -> str:
    return canonical_digest({
        "schema": "gkm.bongard-atomic-smoke-seed-rank.v1",
        "selection_policy": ATOMIC_SMOKE_SELECTION_POLICY,
        "post_freeze_seed": seed, "task_id": task_id,
    })


def atomic_smoke_protocol_digest() -> str:
    """Return the static protocol identity bound into every precommit."""

    return canonical_digest(
        {
            "schema": "gkm.bongard-atomic-smoke-protocol.v2",
            "selection_schema": ATOMIC_SMOKE_SELECTION_SCHEMA,
            "precommit_schema": ATOMIC_SMOKE_PRECOMMIT_SCHEMA,
            "selection_policy": ATOMIC_SMOKE_SELECTION_POLICY,
            "sample_size": ATOMIC_SMOKE_SAMPLE_SIZE,
            "episode_protocol": benchmark.PROTOCOL_VERSION,
            "episode_shape": "6+6-labelled-support-plus-two-unlabelled-queries/v1",
            "causal_order": (
                "authenticate-full-manifest-then-metadata-selection-then-freeze-"
                "canonical-task-paths-then-exposure-record-then-owned-exclusive-"
                "fsync-reload-then-selected-panel-hash-then-episode-prepare/v2"
            ),
            "official_corpus_manifest_digest": OFFICIAL_CORPUS_MANIFEST_DIGEST,
            "official_a3_successor_ledger_digest": (
                OFFICIAL_A3_SUCCESSOR_LEDGER_DIGEST
            ),
            "official_split_source_digest": OFFICIAL_SPLIT_SOURCE_DIGEST,
            "official_historical_seed_digest": OFFICIAL_HISTORICAL_SEED_DIGEST,
            "official_resolver_policy_digest": OFFICIAL_RESOLVER_POLICY_DIGEST,
            "official_blocked_morphology_policy_digest": (
                OFFICIAL_BLOCKED_MORPHOLOGY_POLICY_DIGEST
            ),
            "official_release_descriptor_digest": (
                OFFICIAL_RELEASE_DESCRIPTOR_DIGEST
            ),
            "pre_query_serialization": "query-enumerable-bindings-redacted/v1",
            "transports_or_model_calls": False,
            "dependence_design_authorized": False,
            "calibration_authorized": False,
            "benchmark_claim_authorized": False,
            "official_test_authorized": False,
        }
    )


@dataclass(frozen=True, slots=True)
class ExposurePersistenceReceipt:
    """Immutable receipt for the core-owned durable ledger write."""

    ledger_digest: str
    event_digest: str
    filename: str
    payload_sha256: str
    payload_bytes: int
    receipt_digest: str
    protocol: str = "exclusive-create-or-identical-fsync-file-dir-reload/v1"

    def __post_init__(self) -> None:
        _address(self.ledger_digest, "persisted ledger digest")
        _address(self.event_digest, "persisted event digest")
        _digest(self.payload_sha256, "persisted payload digest")
        _exact_int(self.payload_bytes, "persisted payload byte count", minimum=1)
        expected_name = self.ledger_digest.removeprefix("sha256:") + ".exposure.json"
        if self.filename != expected_name or Path(self.filename).name != self.filename:
            raise AtomicSmokePrecommitError("persistence receipt filename is not canonical")
        if self.protocol != "exclusive-create-or-identical-fsync-file-dir-reload/v1":
            raise AtomicSmokePrecommitError("persistence protocol differs")
        if self.receipt_digest != _content_address(self.content_data()):
            raise AtomicSmokePrecommitError(
                "persistence receipt digest differs from its exact preimage"
            )

    @classmethod
    def create(
        cls, *, ledger: ExposureLedger, filename: str, payload: bytes
    ) -> "ExposurePersistenceReceipt":
        values = {
            "ledger_digest": ledger.digest,
            "event_digest": ledger.events[-1].digest,
            "filename": filename,
            "payload_sha256": hashlib.sha256(payload).hexdigest(),
            "payload_bytes": len(payload),
            "protocol": "exclusive-create-or-identical-fsync-file-dir-reload/v1",
        }
        return cls(**values, receipt_digest=_content_address({
            "schema": ATOMIC_SMOKE_PERSISTENCE_SCHEMA, **values,
        }))

    def content_data(self) -> dict[str, object]:
        return {
            "schema": ATOMIC_SMOKE_PERSISTENCE_SCHEMA,
            "ledger_digest": self.ledger_digest,
            "event_digest": self.event_digest,
            "filename": self.filename,
            "payload_sha256": self.payload_sha256,
            "payload_bytes": self.payload_bytes,
            "protocol": self.protocol,
        }

    def to_data(self) -> dict[str, object]:
        return {**self.content_data(), "receipt_digest": self.receipt_digest}

    @classmethod
    def from_data(cls, value: Mapping[str, Any]) -> "ExposurePersistenceReceipt":
        names = {item.name for item in fields(cls)}
        data = _mapping(
            value, frozenset({"schema", *names}), "exposure persistence receipt"
        )
        if data["schema"] != ATOMIC_SMOKE_PERSISTENCE_SCHEMA:
            raise AtomicSmokePrecommitError("unsupported persistence receipt schema")
        result = cls(**{name: data[name] for name in names})
        if result.to_data() != _canonical_clone(value, "persistence receipt"):
            raise AtomicSmokePrecommitError("persistence receipt is not canonical")
        return result


@dataclass(frozen=True, slots=True)
class _Universe:
    task_ids: tuple[str, ...] = field(repr=False)
    historical_seed_digest: str
    resolver_policy_digest: str
    blocked_morphology_policy_digest: str
    split_source_digest: str


@dataclass(frozen=True, slots=True)
class _AuthenticatedCorpus:
    task_manifests: Mapping[str, TaskManifest] = field(repr=False)
    task_ids: tuple[str, ...] = field(repr=False)


def _validate_task_manifest_value(task: TaskManifest) -> None:
    if not isinstance(task, TaskManifest):
        raise AtomicSmokePrecommitError("full manifest contains a non-TaskManifest")
    _text(task.task_id, "manifest task ID", limit=256)
    if task.family not in FAMILIES or not task.task_id.startswith(task.family + "_"):
        raise AtomicSmokePrecommitError("full manifest task family is inconsistent")
    if len(task.panels) != 14:
        raise AtomicSmokePrecommitError("full manifest task does not contain 14 panels")
    expected = [
        ("positive", index, "1", f"{index}.png") for index in range(7)
    ] + [
        ("negative", index, "0", f"{index}.png") for index in range(7)
    ]
    for panel, (polarity, index, label, filename) in zip(
        task.panels, expected, strict=True
    ):
        _exact_int(panel.index, "panel index")
        _exact_int(panel.size_bytes, "panel byte count", minimum=1)
        _address(panel.sha256, "panel digest")
        if (
            panel.task_id != task.task_id
            or panel.family != task.family
            or panel.polarity != polarity
            or panel.index != index
            or panel.filename != filename
            or panel.panel_id != f"{task.family}/{task.task_id}/{label}/{filename}"
            or not isinstance(panel.path, Path)
        ):
            raise AtomicSmokePrecommitError(
                "full manifest contains a non-canonical panel entry"
            )
    if task.digest != _content_address(task.content_dict()):
        raise AtomicSmokePrecommitError(
            "full manifest task content differs from its digest"
        )


def _authenticate_full_manifest(
    corpus: ShapeBongardCorpus, full_manifest: CorpusManifest
) -> _AuthenticatedCorpus:
    """Authenticate the exact official 12k inventory without opening pixels."""

    if not isinstance(corpus, ShapeBongardCorpus):
        raise TypeError("corpus must be ShapeBongardCorpus")
    if not isinstance(full_manifest, CorpusManifest):
        raise TypeError("full_corpus_manifest must be CorpusManifest")
    if full_manifest.digest != _content_address(full_manifest.content_dict()):
        raise AtomicSmokePrecommitError(
            "full corpus manifest content differs from its digest"
        )
    if full_manifest.digest != OFFICIAL_CORPUS_MANIFEST_DIGEST:
        raise AtomicSmokePrecommitError(
            "full corpus manifest differs from the official commitment"
        )
    if (
        full_manifest.layout != OFFICIAL_CORPUS_LAYOUT
        or corpus.layout != OFFICIAL_CORPUS_LAYOUT
        or corpus.layout != full_manifest.layout
    ):
        raise AtomicSmokePrecommitError("corpus layout differs from the official manifest")

    expected_family = dict(OFFICIAL_FAMILY_COUNTS)
    for value in expected_family.values():
        _exact_int(value, "official family count")
    manifest_family = dict(full_manifest.family_counts)
    corpus_family = dict(corpus.family_counts)
    if any(type(value) is not int for value in manifest_family.values()):
        raise AtomicSmokePrecommitError("manifest family counts are not exact integers")
    if manifest_family != expected_family or corpus_family != expected_family:
        raise AtomicSmokePrecommitError("corpus family counts differ from official counts")
    expected_total = sum(expected_family.values())
    _exact_int(OFFICIAL_TASK_COUNT, "official task count", minimum=1)
    if expected_total != OFFICIAL_TASK_COUNT:
        raise AtomicSmokePrecommitError("official family counts do not sum to task count")
    if len(full_manifest.tasks) != expected_total or len(corpus.tasks) != expected_total:
        raise AtomicSmokePrecommitError("corpus inventory count differs from official counts")

    task_ids = tuple(task.task_id for task in full_manifest.tasks)
    if task_ids != tuple(sorted(task_ids)) or len(set(task_ids)) != len(task_ids):
        raise AtomicSmokePrecommitError("full manifest inventory is not unique and sorted")
    trusted: dict[str, TaskManifest] = {}
    actual_family = Counter()
    for task in full_manifest.tasks:
        _validate_task_manifest_value(task)
        trusted[task.task_id] = task
        actual_family[task.family] += 1
    if dict(actual_family) != {key: value for key, value in expected_family.items() if value}:
        raise AtomicSmokePrecommitError("full manifest task families differ from counts")
    if corpus.task_ids != task_ids:
        raise AtomicSmokePrecommitError("live corpus inventory differs from full manifest")
    if any(
        live.family != trusted[live.task_id].family for live in corpus.tasks
    ):
        raise AtomicSmokePrecommitError("live corpus task families differ from manifest")

    if (
        full_manifest.split.source_digest != OFFICIAL_SPLIT_SOURCE_DIGEST
        or corpus.split.source_digest != OFFICIAL_SPLIT_SOURCE_DIGEST
        or full_manifest.split.to_manifest_dict() != corpus.split.to_manifest_dict()
    ):
        raise AtomicSmokePrecommitError("corpus split differs from official split")
    try:
        full_manifest.split.validate(task_ids, official_counts=False)
    except Exception as exc:
        raise AtomicSmokePrecommitError("full manifest split is incomplete") from exc
    groups = full_manifest.split.canonical_groups
    expected_split = dict(OFFICIAL_SPLIT_COUNTS)
    expected_regime = dict(OFFICIAL_REGIME_COUNTS)
    if (
        {name: len(groups[name]) for name in expected_split} != expected_split
        or {name: len(groups[name]) for name in expected_regime} != expected_regime
    ):
        raise AtomicSmokePrecommitError("official split or regime counts differ")
    return _AuthenticatedCorpus(
        task_manifests=MappingProxyType(trusted), task_ids=task_ids
    )


def _derive_official_universe(
    corpus: ShapeBongardCorpus,
    exposure_ledger: ExposureLedger,
    authenticated: _AuthenticatedCorpus,
    *,
    source_corpus_manifest_digest: str,
    expected_exposure_ledger_digest: str,
) -> _Universe:
    """Derive and authenticate the ten-ID universe without task materialization."""

    if not isinstance(exposure_ledger, ExposureLedger):
        raise TypeError("exposure_ledger must be ExposureLedger")
    source_digest = _address(source_corpus_manifest_digest, "source corpus manifest digest")
    expected_ledger = _address(expected_exposure_ledger_digest, "expected ledger digest")
    if source_digest != OFFICIAL_CORPUS_MANIFEST_DIGEST:
        raise AtomicSmokePrecommitError("source corpus digest is not the official pin")
    if expected_ledger != OFFICIAL_A3_SUCCESSOR_LEDGER_DIGEST:
        raise AtomicSmokePrecommitError("expected predecessor is not the official A3 pin")
    if exposure_ledger.digest != expected_ledger:
        raise AtomicSmokePrecommitError("exposure ledger differs from pinned predecessor")
    if exposure_ledger.corpus_digest != source_digest:
        raise AtomicSmokePrecommitError("exposure predecessor belongs to another corpus")
    if not corpus.split.groups or corpus.split.source_digest is None:
        raise AtomicSmokePrecommitError("selection requires an authenticated split index")
    split_source_digest = _address(corpus.split.source_digest, "split source digest")
    if split_source_digest != OFFICIAL_SPLIT_SOURCE_DIGEST:
        raise AtomicSmokePrecommitError("split source differs from the official pin")
    if not exposure_ledger.exposed_task_ids <= set(authenticated.task_ids):
        raise AtomicSmokePrecommitError(
            "exposure predecessor contains IDs outside the official inventory"
        )

    try:
        historical = load_historical_exposure()
        resolver_digest = semantic_resolver_policy_digest(historical)
        if historical.seed_digest != OFFICIAL_HISTORICAL_SEED_DIGEST:
            raise AtomicSmokePrecommitError("historical seed differs from official pin")
        if resolver_digest != OFFICIAL_RESOLVER_POLICY_DIGEST:
            raise AtomicSmokePrecommitError("semantic resolver differs from official pin")
        blocked_clusters = tuple(
            sorted(
                key.concepts[0]
                for key in semantic_policy_blocked_keys(historical)
                if key.kind == "basic_morphology_cluster"
            )
        )
        blocked = set(blocked_clusters)
        exposed = tuple(
            parse_official_task_id(task_id, historical)
            for task_id in sorted(exposure_ledger.exposed_task_ids)
            if task_id.startswith("bd_")
        )
        if any(item.family != "bd" for item in exposed):
            raise AtomicSmokePrecommitError("a BD exposure parsed under another family")
        exposed_clusters = {
            semantic_generator_cluster_id(item.family, item.concepts) for item in exposed
        }
        report = build_cohort_report(
            corpus, historical, split="train", family="bd", cohort="drill")
        exposed_ids = exposure_ledger.exposed_task_ids
        task_ids = tuple(
            sorted(
                record.task_id
                for record in report.records
                if record.historically_clean
                and record.semantic_cohort == "drill"
                and record.split == "train"
                and record.task_id not in exposed_ids
                and not any(
                    basic_morphology_cluster_id(concept) in blocked
                    for concept in record.parsed.concepts
                )
                and semantic_generator_cluster_id(
                    record.family, record.parsed.concepts
                )
                in exposed_clusters
            )
        )
    except AtomicSmokePrecommitError:
        raise
    except Exception as exc:
        raise AtomicSmokePrecommitError(
            "cannot certify atomic smoke universe from frozen metadata") from exc

    universe_digest = _content_address(list(task_ids))
    if (
        len(task_ids) != OFFICIAL_A3_SUCCESSOR_UNIVERSE_COUNT
        or universe_digest != OFFICIAL_A3_SUCCESSOR_UNIVERSE_DIGEST
    ):
        raise AtomicSmokePrecommitError(
            "atomic smoke universe differs from the official A3 successor "
            f"commitment (count={len(task_ids)}, digest={universe_digest})"
        )
    blocked_policy_digest = _content_address(
        {
            "schema": "gkm.bongard-basic-morphology-block-policy.v1",
            "resolver_policy_digest": resolver_digest,
            "blocked_clusters": list(blocked_clusters),
        }
    )
    if blocked_policy_digest != OFFICIAL_BLOCKED_MORPHOLOGY_POLICY_DIGEST:
        raise AtomicSmokePrecommitError(
            "blocked morphology policy differs from official pin"
        )
    return _Universe(
        task_ids=task_ids,
        historical_seed_digest=historical.seed_digest,
        resolver_policy_digest=resolver_digest,
        blocked_morphology_policy_digest=blocked_policy_digest,
        split_source_digest=split_source_digest,
    )


@dataclass(frozen=True, slots=True)
class AtomicSmokeSelection:
    """Content-addressed metadata-only N=1 selection commitment."""

    source_corpus_manifest_digest: str
    split_source_digest: str
    exposure_predecessor_digest: str
    historical_seed_digest: str
    resolver_policy_digest: str
    blocked_morphology_policy_digest: str
    selection_seed_commitment: str
    selected_task_id: str
    selected_generator_cluster_id: str
    selection_digest: str
    selection_policy: str = ATOMIC_SMOKE_SELECTION_POLICY
    sample_size: int = ATOMIC_SMOKE_SAMPLE_SIZE
    family: str = "bd"
    split: str = "train"
    semantic_cohort: str = "drill"
    universe_count: int = OFFICIAL_A3_SUCCESSOR_UNIVERSE_COUNT
    universe_task_ids_digest: str = OFFICIAL_A3_SUCCESSOR_UNIVERSE_DIGEST
    dependence_design_authorized: bool = False
    calibration_authorized: bool = False
    benchmark_claim_authorized: bool = False
    official_test_authorized: bool = False

    def __post_init__(self) -> None:
        for name in (
            "source_corpus_manifest_digest",
            "split_source_digest",
            "exposure_predecessor_digest",
            "historical_seed_digest",
            "resolver_policy_digest",
            "blocked_morphology_policy_digest",
            "universe_task_ids_digest",
            "selection_digest",
        ):
            _address(getattr(self, name), name.replace("_", " "))
        _digest(self.selection_seed_commitment, "selection seed commitment")
        _exact_int(self.sample_size, "sample size", minimum=1)
        _exact_int(self.universe_count, "universe count", minimum=1)
        if (
            self.selection_policy != ATOMIC_SMOKE_SELECTION_POLICY
            or self.sample_size != ATOMIC_SMOKE_SAMPLE_SIZE
            or self.family != "bd"
            or self.split != "train"
            or self.semantic_cohort != "drill"
            or self.universe_count != OFFICIAL_A3_SUCCESSOR_UNIVERSE_COUNT
            or self.universe_task_ids_digest
            != OFFICIAL_A3_SUCCESSOR_UNIVERSE_DIGEST
            or self.source_corpus_manifest_digest
            != OFFICIAL_CORPUS_MANIFEST_DIGEST
            or self.split_source_digest != OFFICIAL_SPLIT_SOURCE_DIGEST
            or self.exposure_predecessor_digest
            != OFFICIAL_A3_SUCCESSOR_LEDGER_DIGEST
            or self.historical_seed_digest != OFFICIAL_HISTORICAL_SEED_DIGEST
            or self.resolver_policy_digest != OFFICIAL_RESOLVER_POLICY_DIGEST
            or self.blocked_morphology_policy_digest
            != OFFICIAL_BLOCKED_MORPHOLOGY_POLICY_DIGEST
        ):
            raise AtomicSmokePrecommitError(
                "atomic smoke selection scope or official universe differs"
            )
        if any(
            getattr(self, name) is not False
            for name in (
                "dependence_design_authorized",
                "calibration_authorized",
                "benchmark_claim_authorized",
                "official_test_authorized",
            )
        ):
            raise AtomicSmokePrecommitError(
                "atomic smoke selection cannot grant scientific authorization"
            )
        _text(self.selected_task_id, "selected task ID", limit=256)
        _text(
            self.selected_generator_cluster_id,
            "selected generator cluster ID",
            limit=256,
        )
        try:
            parsed = parse_official_task_id(self.selected_task_id)
            expected_cluster = semantic_generator_cluster_id(
                parsed.family, parsed.concepts
            )
        except Exception as exc:
            raise AtomicSmokePrecommitError(
                "selected task is not a canonical official BD identity"
            ) from exc
        if parsed.family != "bd" or expected_cluster != (
            self.selected_generator_cluster_id
        ):
            raise AtomicSmokePrecommitError(
                "selected task and generator cluster differ"
            )
        if self.selection_digest != _content_address(self.content_data()):
            raise AtomicSmokePrecommitError(
                "selection digest differs from its exact preimage"
            )

    @property
    def digest(self) -> str:
        return self.selection_digest

    @classmethod
    def create(
        cls,
        *,
        source_corpus_manifest_digest: str,
        split_source_digest: str,
        exposure_predecessor_digest: str,
        historical_seed_digest: str,
        resolver_policy_digest: str,
        blocked_morphology_policy_digest: str,
        seed: str,
        selected_task_id: str,
        selected_generator_cluster_id: str,
    ) -> "AtomicSmokeSelection":
        values = {
            "source_corpus_manifest_digest": source_corpus_manifest_digest,
            "split_source_digest": split_source_digest,
            "exposure_predecessor_digest": exposure_predecessor_digest,
            "historical_seed_digest": historical_seed_digest,
            "resolver_policy_digest": resolver_policy_digest,
            "blocked_morphology_policy_digest": blocked_morphology_policy_digest,
            "selection_seed_commitment": _seed_commitment(seed),
            "selected_task_id": selected_task_id,
            "selected_generator_cluster_id": selected_generator_cluster_id,
            "selection_policy": ATOMIC_SMOKE_SELECTION_POLICY,
            "sample_size": 1,
            "family": "bd",
            "split": "train",
            "semantic_cohort": "drill",
            "universe_count": OFFICIAL_A3_SUCCESSOR_UNIVERSE_COUNT,
            "universe_task_ids_digest": OFFICIAL_A3_SUCCESSOR_UNIVERSE_DIGEST,
            "dependence_design_authorized": False,
            "calibration_authorized": False,
            "benchmark_claim_authorized": False,
            "official_test_authorized": False,
        }
        content = {"schema": ATOMIC_SMOKE_SELECTION_SCHEMA, **values}
        return cls(**values, selection_digest=_content_address(content))  # type: ignore[arg-type]

    def content_data(self) -> dict[str, object]:
        return {
            "schema": ATOMIC_SMOKE_SELECTION_SCHEMA,
            **{
                item.name: getattr(self, item.name)
                for item in fields(self)
                if item.name != "selection_digest"
            },
        }

    def to_data(self) -> dict[str, object]:
        return {**self.content_data(), "selection_digest": self.selection_digest}

    @classmethod
    def from_data(cls, value: Mapping[str, Any]) -> "AtomicSmokeSelection":
        field_names = {item.name for item in fields(cls)}
        data = _mapping(
            value,
            frozenset({"schema", *field_names}),
            "atomic smoke selection",
        )
        if data["schema"] != ATOMIC_SMOKE_SELECTION_SCHEMA:
            raise AtomicSmokePrecommitError(
                "unsupported atomic smoke selection schema"
            )
        result = cls(**{name: data[name] for name in field_names})  # type: ignore[arg-type]
        if result.to_data() != _canonical_clone(value, "selection"):
            raise AtomicSmokePrecommitError(
                "atomic smoke selection is not canonical"
            )
        return result


def select_atomic_smoke_task(
    corpus: ShapeBongardCorpus,
    *,
    seed: str,
    full_corpus_manifest: CorpusManifest,
    source_corpus_manifest_digest: str,
    exposure_ledger: ExposureLedger,
    expected_exposure_ledger_digest: str,
) -> AtomicSmokeSelection:
    """Select the seed-minimum task after freezing the exact ten-ID universe."""

    _text(seed, "post-freeze selection seed")
    authenticated = _authenticate_full_manifest(corpus, full_corpus_manifest)
    universe = _derive_official_universe(
        corpus,
        exposure_ledger,
        authenticated,
        source_corpus_manifest_digest=source_corpus_manifest_digest,
        expected_exposure_ledger_digest=expected_exposure_ledger_digest,
    )
    selected_task_id = min(
        universe.task_ids,
        key=lambda task_id: (_seed_rank(seed, task_id), task_id),
    )
    parsed = parse_official_task_id(selected_task_id)
    return AtomicSmokeSelection.create(
        source_corpus_manifest_digest=source_corpus_manifest_digest,
        split_source_digest=universe.split_source_digest,
        exposure_predecessor_digest=exposure_ledger.digest,
        historical_seed_digest=universe.historical_seed_digest,
        resolver_policy_digest=universe.resolver_policy_digest,
        blocked_morphology_policy_digest=(
            universe.blocked_morphology_policy_digest
        ),
        seed=seed,
        selected_task_id=selected_task_id,
        selected_generator_cluster_id=semantic_generator_cluster_id(
            parsed.family, parsed.concepts
        ),
    )


def replay_atomic_smoke_selection(
    value: AtomicSmokeSelection | Mapping[str, Any],
    *,
    expected_selection_digest: str,
    corpus: ShapeBongardCorpus,
    seed: str,
    full_corpus_manifest: CorpusManifest,
    source_corpus_manifest_digest: str,
    exposure_ledger: ExposureLedger,
) -> AtomicSmokeSelection:
    """Cold-rederive a selection under external digest, seed, and source pins."""

    expected = _address(
        expected_selection_digest, "expected selection digest"
    )
    archived = (
        AtomicSmokeSelection.from_data(value)
        if isinstance(value, Mapping)
        else AtomicSmokeSelection.from_data(value.to_data())
    )
    if archived.digest != expected:
        raise AtomicSmokePrecommitError(
            "decoded selection differs from expected selection digest"
        )
    if archived.source_corpus_manifest_digest != _address(
        source_corpus_manifest_digest, "source corpus manifest digest"
    ):
        raise AtomicSmokePrecommitError(
            "selection differs from the externally pinned source corpus"
        )
    replayed = select_atomic_smoke_task(
        corpus,
        seed=seed,
        full_corpus_manifest=full_corpus_manifest,
        source_corpus_manifest_digest=source_corpus_manifest_digest,
        exposure_ledger=exposure_ledger,
        expected_exposure_ledger_digest=archived.exposure_predecessor_digest,
    )
    if replayed.to_data() != archived.to_data():
        raise AtomicSmokePrecommitError(
            "atomic smoke selection differs from independent replay"
        )
    return replayed


@dataclass(frozen=True, slots=True)
class _FrozenSelectedTask:
    task: BongardTask = field(repr=False)
    trusted_manifest: TaskManifest = field(repr=False)
    split: object = field(repr=False)
    corpus_root: Path


def _freeze_selected_task_binding(
    corpus: ShapeBongardCorpus,
    authenticated: _AuthenticatedCorpus,
    task_id: str,
) -> _FrozenSelectedTask:
    """Freeze canonical selected paths without reading or hashing panel bytes."""

    matches = tuple(task for task in corpus.tasks if task.task_id == task_id)
    if len(matches) != 1:
        raise AtomicSmokePrecommitError("selected task is not unique in live inventory")
    task = matches[0]
    if corpus.task(task_id) is not task:
        raise AtomicSmokePrecommitError("corpus task lookup redirects selected ownership")
    try:
        corpus_root = corpus.root.resolve(strict=True)
    except OSError as exc:
        raise AtomicSmokePrecommitError("corpus root cannot be canonicalized") from exc
    if corpus.root != corpus_root:
        raise AtomicSmokePrecommitError("corpus root is not a canonical owned path")
    component = "images" if corpus.layout == "archive" else "png"
    expected_root = corpus_root / task.family / component / task_id
    try:
        resolved_root = task.root.resolve(strict=True)
    except OSError as exc:
        raise AtomicSmokePrecommitError("selected task root cannot be canonicalized") from exc
    if task.root != expected_root or resolved_root != expected_root:
        raise AtomicSmokePrecommitError("selected task root escapes canonical ownership")

    expected_paths: list[Path] = []
    for label in ("1", "0"):
        for index in range(7):
            path = expected_root / label / f"{index}.png"
            try:
                resolved = path.resolve(strict=True)
            except OSError as exc:
                raise AtomicSmokePrecommitError(
                    "selected panel path cannot be canonicalized"
                ) from exc
            if resolved != path or not path.is_file():
                raise AtomicSmokePrecommitError(
                    "selected panel path escapes canonical task ownership"
                )
            expected_paths.append(path)
    if (*task.positive, *task.negative) != tuple(expected_paths):
        raise AtomicSmokePrecommitError("selected task panel paths were redirected")
    trusted = authenticated.task_manifests[task_id]
    if tuple(panel.path for panel in trusted.panels) != tuple(expected_paths):
        raise AtomicSmokePrecommitError(
            "trusted selected manifest paths differ from canonical ownership"
        )
    return _FrozenSelectedTask(
        task=task,
        trusted_manifest=trusted,
        split=corpus.split,
        corpus_root=corpus_root,
    )


def _assert_frozen_binding(
    corpus: ShapeBongardCorpus, frozen: _FrozenSelectedTask
) -> None:
    if (
        corpus.root != frozen.corpus_root
        or corpus.split is not frozen.split
        or corpus.task(frozen.task.task_id) is not frozen.task
        or tuple(task for task in corpus.tasks if task.task_id == frozen.task.task_id)
        != (frozen.task,)
    ):
        raise AtomicSmokePrecommitError(
            "selected task ownership changed across persistence"
        )


def _development_manifest(
    corpus: ShapeBongardCorpus, task_manifest: TaskManifest
) -> CorpusManifest:
    """Hash exactly one selected development task."""

    counts = Counter(("bd",))
    provisional = CorpusManifest(
        layout=corpus.layout,
        family_counts=tuple(
            (family, counts.get(family, 0)) for family in FAMILIES
        ),
        tasks=(task_manifest,),
        split=corpus.split,
        digest="sha256:" + "0" * 64,
    )
    return CorpusManifest(
        layout=provisional.layout,
        family_counts=provisional.family_counts,
        tasks=provisional.tasks,
        split=provisional.split,
        digest=_content_address(provisional.content_dict()),
    )


_MAX_EXPOSURE_LEDGER_BYTES = 64 * 1024 * 1024


def _read_stable_exposure_file(path: Path) -> bytes:
    """Read one bounded regular-file inode and retain its pathname binding."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        before_path = os.lstat(path)
        if (
            not stat.S_ISREG(before_path.st_mode)
            or not 0 < before_path.st_size <= _MAX_EXPOSURE_LEDGER_BYTES
        ):
            raise OSError("exposure path is not a bounded regular file")
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise AtomicSmokePrecommitError(
            "cannot open stable no-follow exposure successor"
        ) from exc
    identity = (
        before_path.st_dev,
        before_path.st_ino,
        before_path.st_size,
        before_path.st_mtime_ns,
        before_path.st_ctime_ns,
    )
    chunks: list[bytes] = []
    total = 0
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
            opened.st_mtime_ns,
            opened.st_ctime_ns,
        ) != identity:
            raise OSError("exposure path changed while being opened")
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > _MAX_EXPOSURE_LEDGER_BYTES:
                raise OSError("exposure successor became oversized")
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if (
            total != opened.st_size
            or not stat.S_ISREG(after.st_mode)
            or (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            )
            != identity
        ):
            raise OSError("exposure successor changed while being read")
    except OSError as exc:
        raise AtomicSmokePrecommitError(
            "cannot read stable no-follow exposure successor"
        ) from exc
    finally:
        os.close(descriptor)
    try:
        after_path = os.lstat(path)
    except OSError as exc:
        raise AtomicSmokePrecommitError(
            "exposure successor path changed after reading"
        ) from exc
    if (
        not stat.S_ISREG(after_path.st_mode)
        or (
            after_path.st_dev,
            after_path.st_ino,
            after_path.st_size,
            after_path.st_mtime_ns,
            after_path.st_ctime_ns,
        )
        != identity
    ):
        raise AtomicSmokePrecommitError(
            "exposure successor path changed while being read"
        )
    return b"".join(chunks)


def _decode_exposure_payload(payload: bytes) -> ExposureLedger:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        raw = json.loads(
            payload.decode("utf-8", errors="strict"),
            object_pairs_hook=unique_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON token {token}")
            ),
        )
        if not isinstance(raw, dict):
            raise ValueError("exposure successor JSON is not an object")
        return ExposureLedger.from_dict(raw)
    except Exception as exc:
        raise AtomicSmokePrecommitError(
            "cannot decode durable exposure successor"
        ) from exc


def _persist_exposure_successor(
    successor: ExposureLedger, exposure_store_dir: str | Path
) -> tuple[ExposureLedger, ExposurePersistenceReceipt]:
    """Durably create or verify one content-addressed successor and reload it."""

    directory = Path(exposure_store_dir)
    try:
        directory = directory.resolve(strict=True)
    except OSError as exc:
        raise AtomicSmokePrecommitError(
            "exposure store directory must already exist"
        ) from exc
    if not directory.is_dir():
        raise AtomicSmokePrecommitError("exposure store path is not a directory")
    payload = successor.to_json().encode("utf-8")
    filename = successor.digest.removeprefix("sha256:") + ".exposure.json"
    destination = directory / filename
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    created = False
    try:
        descriptor = os.open(destination, flags, 0o600)
    except FileExistsError:
        existing = _read_stable_exposure_file(destination)
        if existing != payload:
            raise AtomicSmokePrecommitError(
                "content-addressed exposure path contains different bytes"
            )
    except OSError as exc:
        raise AtomicSmokePrecommitError(
            "cannot exclusively create exposure successor"
        ) from exc
    else:
        created = True
        try:
            view = memoryview(payload)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError("short exposure ledger write")
                view = view[written:]
            os.fsync(descriptor)
        except OSError as exc:
            raise AtomicSmokePrecommitError(
                "cannot durably write exposure successor"
            ) from exc
        finally:
            os.close(descriptor)
    try:
        dir_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        directory_fd = os.open(directory, dir_flags)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except OSError as exc:
        boundary = "new" if created else "existing"
        raise AtomicSmokePrecommitError(
            f"cannot fsync exposure directory after {boundary} successor"
        ) from exc
    persisted_payload = _read_stable_exposure_file(destination)
    if persisted_payload != payload:
        raise AtomicSmokePrecommitError(
            "durable exposure bytes differ after fsync"
        )
    reloaded = _decode_exposure_payload(persisted_payload)
    if reloaded.to_dict() != successor.to_dict():
        raise AtomicSmokePrecommitError(
            "reloaded exposure successor differs from exact persisted value"
        )
    receipt = ExposurePersistenceReceipt.create(
        ledger=reloaded, filename=filename, payload=payload
    )
    return reloaded, receipt


def _verify_persisted_receipt(
    receipt: ExposurePersistenceReceipt,
    successor: ExposureLedger,
    exposure_store_dir: str | Path,
) -> None:
    try:
        directory = Path(exposure_store_dir).resolve(strict=True)
        destination = directory / receipt.filename
        payload = _read_stable_exposure_file(destination)
        reloaded = _decode_exposure_payload(payload)
    except Exception as exc:
        raise AtomicSmokePrecommitError(
            "cannot verify persisted exposure receipt"
        ) from exc
    if (
        len(payload) != receipt.payload_bytes
        or hashlib.sha256(payload).hexdigest() != receipt.payload_sha256
        or reloaded.to_dict() != successor.to_dict()
        or reloaded.digest != receipt.ledger_digest
    ):
        raise AtomicSmokePrecommitError(
            "persisted exposure bytes differ from bound receipt"
        )


def _manifest_public_data(
    manifest: CorpusManifest, selection: AtomicSmokeSelection
) -> dict[str, object]:
    if len(manifest.tasks) != 1:
        raise AtomicSmokePrecommitError(
            "atomic smoke development manifest must contain one task"
        )
    task = manifest.tasks[0]
    return {
        "schema": ATOMIC_SMOKE_DEVELOPMENT_MANIFEST_SCHEMA,
        "layout": manifest.layout,
        "family_counts": dict(manifest.family_counts),
        "task_id": selection.selected_task_id,
        "task_manifest_digest": task.digest,
        "split_source_digest": selection.split_source_digest,
        "development_manifest_digest": manifest.digest,
    }


_MANIFEST_FIELDS = frozenset({
    "schema", "layout", "family_counts", "task_id", "task_manifest_digest",
    "split_source_digest", "development_manifest_digest",
})
_EPISODE_FIELDS = frozenset({
    "version", "task_id", "family", "split", "regime", "run_id",
    "verifier_id", "seed_digest", "corpus_digest", "task_manifest_digest",
    "support_commitment_digest", "label_commitment_digest",
})


def _episode_public_projection(plan: benchmark.EpisodePlan) -> dict[str, object]:
    data = plan.to_data()
    data.pop("latent_query_digest", None)
    if set(data) != _EPISODE_FIELDS:
        raise AtomicSmokePrecommitError(
            "episode plan acquired an unreviewed pre-query public field"
        )
    return data


def _validate_manifest_public_data(
    value: object, selection: AtomicSmokeSelection
) -> Mapping[str, Any]:
    data = _mapping(value, _MANIFEST_FIELDS, "development manifest public data")
    family_counts = data["family_counts"]
    if (
        data["schema"] != ATOMIC_SMOKE_DEVELOPMENT_MANIFEST_SCHEMA
        or data["task_id"] != selection.selected_task_id
        or data["split_source_digest"] != selection.split_source_digest
        or not isinstance(family_counts, Mapping)
        or set(family_counts) != {"ff", "bd", "hd"}
        or any(type(value) is not int for value in family_counts.values())
        or dict(family_counts) != {"ff": 0, "bd": 1, "hd": 0}
        or not isinstance(data["layout"], str)
        or not data["layout"]
    ):
        raise AtomicSmokePrecommitError("development manifest public scope differs")
    _address(data["task_manifest_digest"], "task manifest digest")
    _address(data["development_manifest_digest"], "development manifest digest")
    return data


def _validate_episode_public_data(
    value: object,
    selection: AtomicSmokeSelection,
    manifest_data: Mapping[str, Any],
) -> Mapping[str, Any]:
    data = _mapping(value, _EPISODE_FIELDS, "episode public data")
    if (
        data["version"] != benchmark.PROTOCOL_VERSION
        or data["task_id"] != selection.selected_task_id
        or data["family"] != "bd"
        or data["split"] != "train"
        or data["regime"] is not None
        or not isinstance(data["run_id"], str)
        or not data["run_id"].startswith("run-")
        or not isinstance(data["verifier_id"], str)
        or not data["verifier_id"].strip()
    ):
        raise AtomicSmokePrecommitError("episode public scope differs")
    for name in (
        "seed_digest",
        "corpus_digest",
        "task_manifest_digest",
        "support_commitment_digest",
        "label_commitment_digest",
    ):
        _digest(data[name], name.replace("_", " "))
    if data["corpus_digest"] != str(
        manifest_data["development_manifest_digest"]
    ).removeprefix("sha256:") or data["task_manifest_digest"] != str(
        manifest_data["task_manifest_digest"]
    ).removeprefix("sha256:"):
        raise AtomicSmokePrecommitError(
            "episode does not descend from the development manifest")
    return data


@dataclass(frozen=True, slots=True)
class AtomicSmokePrecommit:
    """Public precommit plus optional live private planning state."""

    selection: AtomicSmokeSelection
    exposure_predecessor_digest: str
    exposure_successor_digest: str
    exposure_event_digest: str
    exposure_persistence_receipt: ExposurePersistenceReceipt
    source_dependency_digest: str
    protocol_digest: str
    development_manifest_data: Mapping[str, Any]
    development_manifest_digest: str
    episode_public_data: Mapping[str, Any]
    episode_digest: str
    precommit_digest: str
    _episode_plan: benchmark.EpisodePlan | None = field(
        default=None, repr=False, compare=False
    )
    _exposure_successor: ExposureLedger | None = field(
        default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if not isinstance(self.selection, AtomicSmokeSelection):
            raise TypeError("selection must be AtomicSmokeSelection")
        for name in (
            "exposure_predecessor_digest",
            "exposure_successor_digest",
            "exposure_event_digest",
            "development_manifest_digest",
            "precommit_digest",
        ):
            _address(getattr(self, name), name.replace("_", " "))
        _digest(self.source_dependency_digest, "source dependency digest")
        _digest(self.protocol_digest, "protocol digest")
        _digest(self.episode_digest, "episode digest")
        if self.protocol_digest != atomic_smoke_protocol_digest():
            raise AtomicSmokePrecommitError("atomic smoke protocol digest differs")
        if self.exposure_predecessor_digest != self.selection.exposure_predecessor_digest:
            raise AtomicSmokePrecommitError("precommit predecessor differs from selection")
        if not isinstance(
            self.exposure_persistence_receipt, ExposurePersistenceReceipt
        ):
            raise TypeError("exposure persistence receipt has the wrong type")
        if (
            self.exposure_persistence_receipt.ledger_digest
            != self.exposure_successor_digest
            or self.exposure_persistence_receipt.event_digest
            != self.exposure_event_digest
        ):
            raise AtomicSmokePrecommitError(
                "persistence receipt differs from exposure successor"
            )
        manifest = _validate_manifest_public_data(
            self.development_manifest_data, self.selection
        )
        if self.development_manifest_digest != manifest["development_manifest_digest"]:
            raise AtomicSmokePrecommitError("development manifest digest parent differs")
        episode = _validate_episode_public_data(
            self.episode_public_data, self.selection, manifest
        )
        if self.episode_digest != canonical_digest(_thaw_json(episode)):
            raise AtomicSmokePrecommitError("episode digest differs from its public data")
        if self.precommit_digest != _content_address(self.content_data()):
            raise AtomicSmokePrecommitError("precommit digest differs from its exact preimage")
        if self._episode_plan is not None:
            if not isinstance(self._episode_plan, benchmark.EpisodePlan):
                raise TypeError("private episode plan must be EpisodePlan or None")
            if _episode_public_projection(self._episode_plan) != _thaw_json(
                self.episode_public_data
            ):
                raise AtomicSmokePrecommitError(
                    "private episode plan differs from public commitment")
        if self._exposure_successor is not None:
            if not isinstance(self._exposure_successor, ExposureLedger):
                raise TypeError("private exposure successor must be ExposureLedger or None")
            if self._exposure_successor.digest != self.exposure_successor_digest:
                raise AtomicSmokePrecommitError(
                    "private exposure successor differs from public commitment")

    @property
    def digest(self) -> str:
        return self.precommit_digest

    @property
    def episode_plan(self) -> benchmark.EpisodePlan:
        """Return live private planning state; decoded archives are detached."""

        if self._episode_plan is None:
            raise AtomicSmokePrecommitError(
                "decoded precommit has no live private episode plan"
            )
        return self._episode_plan

    @property
    def exposure_successor(self) -> ExposureLedger:
        if self._exposure_successor is None:
            raise AtomicSmokePrecommitError(
                "decoded precommit has no live exposure successor"
            )
        return self._exposure_successor

    @classmethod
    def create(
        cls,
        *,
        selection: AtomicSmokeSelection,
        exposure_successor: ExposureLedger,
        exposure_persistence_receipt: ExposurePersistenceReceipt,
        source_dependency_digest: str,
        development_manifest: CorpusManifest,
        episode_plan: benchmark.EpisodePlan,
    ) -> "AtomicSmokePrecommit":
        if not exposure_successor.events:
            raise AtomicSmokePrecommitError(
                "atomic smoke successor has no disclosure event"
            )
        manifest_data = _manifest_public_data(development_manifest, selection)
        episode_data = _episode_public_projection(episode_plan)
        values = {
            "selection": selection.to_data(),
            "exposure_predecessor_digest": selection.exposure_predecessor_digest,
            "exposure_successor_digest": exposure_successor.digest,
            "exposure_event_digest": exposure_successor.events[-1].digest,
            "exposure_persistence_receipt": (
                exposure_persistence_receipt.to_data()
            ),
            "source_dependency_digest": source_dependency_digest,
            "protocol_digest": atomic_smoke_protocol_digest(),
            "development_manifest_data": manifest_data,
            "development_manifest_digest": development_manifest.digest,
            "episode_public_data": episode_data,
            "episode_digest": canonical_digest(episode_data),
        }
        content = {
            "schema": ATOMIC_SMOKE_PRECOMMIT_SCHEMA,
            **values,
            "selection_digest": selection.digest,
        }
        return cls(
            selection=selection,
            exposure_persistence_receipt=exposure_persistence_receipt,
            **{
                key: value
                for key, value in values.items()
                if key not in {
                    "selection", "exposure_persistence_receipt",
                    "development_manifest_data", "episode_public_data",
                }
            },
            development_manifest_data=_freeze_json(manifest_data),  # type: ignore[arg-type]
            episode_public_data=_freeze_json(episode_data),  # type: ignore[arg-type]
            precommit_digest=_content_address(content),
            _episode_plan=episode_plan,
            _exposure_successor=exposure_successor,
        )  # type: ignore[arg-type]

    def content_data(self) -> dict[str, object]:
        return {
            "schema": ATOMIC_SMOKE_PRECOMMIT_SCHEMA,
            "selection_digest": self.selection.digest,
            **{
                item.name: (
                    self.selection.to_data()
                    if item.name == "selection"
                    else self.exposure_persistence_receipt.to_data()
                    if item.name == "exposure_persistence_receipt"
                    else _thaw_json(getattr(self, item.name))
                )
                for item in fields(self)
                if not item.name.startswith("_")
                and item.name != "precommit_digest"
            },
        }

    def to_data(self) -> dict[str, object]:
        return {**self.content_data(), "precommit_digest": self.precommit_digest}

    @classmethod
    def from_data(cls, value: Mapping[str, Any]) -> "AtomicSmokePrecommit":
        public_names = {
            item.name for item in fields(cls) if not item.name.startswith("_")
        }
        data = _mapping(
            value,
            frozenset({"schema", "selection_digest", *public_names}),
            "atomic smoke precommit",
        )
        if data["schema"] != ATOMIC_SMOKE_PRECOMMIT_SCHEMA:
            raise AtomicSmokePrecommitError(
                "unsupported atomic smoke precommit schema"
            )
        if not isinstance(data["selection"], Mapping):
            raise AtomicSmokePrecommitError("selection must be an object")
        selection = AtomicSmokeSelection.from_data(data["selection"])
        if data["selection_digest"] != selection.digest:
            raise AtomicSmokePrecommitError(
                "precommit selection digest parent differs"
            )
        manifest = _canonical_clone(
            data["development_manifest_data"], "development manifest public data"
        )
        episode = _canonical_clone(data["episode_public_data"], "episode public data")
        persistence = ExposurePersistenceReceipt.from_data(
            data["exposure_persistence_receipt"]
        )
        values = {name: data[name] for name in public_names}
        values.update(
            selection=selection,
            exposure_persistence_receipt=persistence,
            development_manifest_data=_freeze_json(manifest),
            episode_public_data=_freeze_json(episode),
        )
        result = cls(**values)  # type: ignore[arg-type]
        if result.to_data() != _canonical_clone(value, "precommit"):
            raise AtomicSmokePrecommitError(
                "atomic smoke precommit is not canonical"
            )
        return result


def _verify_successor(
    predecessor: ExposureLedger,
    successor: ExposureLedger,
    selection: AtomicSmokeSelection,
    *,
    verifier_id: str,
) -> None:
    if successor.corpus_digest != predecessor.corpus_digest:
        raise AtomicSmokePrecommitError(
            "exposure successor belongs to another corpus"
        )
    if len(successor.events) != len(predecessor.events) + 1 or (
        successor.events[:-1] != predecessor.events
    ):
        raise AtomicSmokePrecommitError(
            "exposure successor is not one append after predecessor"
        )
    event = successor.events[-1]
    if (
        event.phase != ATOMIC_SMOKE_EXPOSURE_PHASE
        or event.actor != verifier_id
        or event.purpose != ATOMIC_SMOKE_EXPOSURE_PURPOSE
        or event.task_ids != (selection.selected_task_id,)
        or event.panel_ids
        or event.source != "atomic-smoke-selection:" + selection.digest
    ):
        raise AtomicSmokePrecommitError(
            "exposure successor event differs from atomic smoke protocol"
        )
    predecessor.assert_unseen(task_ids=(selection.selected_task_id,))


def prepare_atomic_smoke_precommit(
    corpus: ShapeBongardCorpus,
    *,
    seed: str,
    episode_seed: str,
    full_corpus_manifest: CorpusManifest,
    source_corpus_manifest_digest: str,
    source_dependency_digest: str,
    exposure_ledger: ExposureLedger,
    expected_exposure_ledger_digest: str,
    label_seal_nonce: str,
    exposure_store_dir: str | Path,
    verifier_id: str = benchmark.DEFAULT_VERIFIER,
    observed_at: str | None = None,
) -> AtomicSmokePrecommit:
    """Own the durable disclosure boundary, then hash and prepare one episode."""

    _digest(source_dependency_digest, "source dependency digest")
    _digest(label_seal_nonce, "private label nonce")
    _digest(episode_seed, "private episode seed")
    if episode_seed == seed:
        raise AtomicSmokePrecommitError(
            "private episode seed must differ from the selection seed"
        )
    _text(verifier_id, "verifier ID", limit=256)
    authenticated = _authenticate_full_manifest(corpus, full_corpus_manifest)
    selection = select_atomic_smoke_task(
        corpus,
        seed=seed,
        full_corpus_manifest=full_corpus_manifest,
        source_corpus_manifest_digest=source_corpus_manifest_digest,
        exposure_ledger=exposure_ledger,
        expected_exposure_ledger_digest=expected_exposure_ledger_digest,
    )
    frozen = _freeze_selected_task_binding(
        corpus, authenticated, selection.selected_task_id
    )
    successor = exposure_ledger.record(
        phase=ATOMIC_SMOKE_EXPOSURE_PHASE,
        actor=verifier_id,
        purpose=ATOMIC_SMOKE_EXPOSURE_PURPOSE,
        task_ids=(selection.selected_task_id,),
        source="atomic-smoke-selection:" + selection.digest,
        observed_at=observed_at,
        known_task_ids=authenticated.task_ids,
        require_unseen=True,
    )
    _verify_successor(
        exposure_ledger, successor, selection, verifier_id=verifier_id
    )
    durable_successor, persistence_receipt = _persist_exposure_successor(
        successor, exposure_store_dir
    )
    if durable_successor.to_dict() != successor.to_dict():
        raise AtomicSmokePrecommitError("durable successor differs after reload")

    # Re-run metadata and ownership commitments after the fsync/reload boundary.
    replayed = select_atomic_smoke_task(
        corpus,
        seed=seed,
        full_corpus_manifest=full_corpus_manifest,
        source_corpus_manifest_digest=source_corpus_manifest_digest,
        exposure_ledger=exposure_ledger,
        expected_exposure_ledger_digest=expected_exposure_ledger_digest,
    )
    if replayed.to_data() != selection.to_data():
        raise AtomicSmokePrecommitError(
            "selection metadata changed across the durability boundary"
        )
    _assert_frozen_binding(corpus, frozen)
    assignment = frozen.split.assignment(selection.selected_task_id)
    if assignment.split != "train" or assignment.regime is not None:
        raise AtomicSmokePrecommitError(
            "selected task is no longer an ordinary train task"
        )

    fresh_task_manifest = frozen.task.build_manifest()
    if fresh_task_manifest != frozen.trusted_manifest:
        raise AtomicSmokePrecommitError(
            "fresh selected TaskManifest differs from trusted full-manifest entry"
        )
    episode_corpus = ShapeBongardCorpus(
        frozen.corpus_root,
        (frozen.task,),
        layout=corpus.layout,
        split=frozen.split,
    )
    development_manifest = _development_manifest(
        episode_corpus, fresh_task_manifest
    )
    episode_plan = benchmark.prepare_episode(
        episode_corpus,
        selection.selected_task_id,
        seed=f"atomic-smoke-private-episode:{episode_seed}",
        corpus_manifest=development_manifest,
        verifier_id=verifier_id,
        label_seal_nonce=label_seal_nonce,
    )
    return AtomicSmokePrecommit.create(
        selection=selection,
        exposure_successor=durable_successor,
        exposure_persistence_receipt=persistence_receipt,
        source_dependency_digest=source_dependency_digest,
        development_manifest=development_manifest,
        episode_plan=episode_plan,
    )


def cold_decode_and_replay_atomic_smoke_precommit(
    value: Mapping[str, Any],
    *,
    expected_precommit_digest: str,
    corpus: ShapeBongardCorpus,
    seed: str,
    episode_seed: str,
    full_corpus_manifest: CorpusManifest,
    source_corpus_manifest_digest: str,
    source_dependency_digest: str,
    exposure_predecessor: ExposureLedger,
    exposure_successor: ExposureLedger,
    exposure_store_dir: str | Path,
    label_seal_nonce: str,
) -> AtomicSmokePrecommit:
    """Cold-reproduce selection, ledger edge, panel manifest, and episode."""

    expected = _address(
        expected_precommit_digest, "expected precommit digest"
    )
    _digest(episode_seed, "private episode seed")
    if episode_seed == seed:
        raise AtomicSmokePrecommitError(
            "private episode seed must differ from the selection seed"
        )
    archived = AtomicSmokePrecommit.from_data(value)
    if archived.digest != expected:
        raise AtomicSmokePrecommitError(
            "decoded precommit differs from expected precommit digest"
        )
    source_dependency = _digest(
        source_dependency_digest, "source dependency digest"
    )
    if archived.source_dependency_digest != source_dependency:
        raise AtomicSmokePrecommitError(
            "precommit differs from externally pinned source dependencies"
        )
    if archived.selection.source_corpus_manifest_digest != _address(
        source_corpus_manifest_digest, "source corpus manifest digest"
    ):
        raise AtomicSmokePrecommitError(
            "precommit differs from externally pinned source corpus"
        )
    selection = replay_atomic_smoke_selection(
        archived.selection,
        expected_selection_digest=archived.selection.digest,
        corpus=corpus,
        seed=seed,
        full_corpus_manifest=full_corpus_manifest,
        source_corpus_manifest_digest=source_corpus_manifest_digest,
        exposure_ledger=exposure_predecessor,
    )
    if exposure_successor.digest != archived.exposure_successor_digest:
        raise AtomicSmokePrecommitError(
            "external exposure successor differs from precommit"
        )
    verifier_id = archived.episode_public_data["verifier_id"]
    if not isinstance(verifier_id, str):  # guarded by decoder; helps typing.
        raise AtomicSmokePrecommitError("episode verifier ID is malformed")
    _verify_successor(
        exposure_predecessor,
        exposure_successor,
        selection,
        verifier_id=verifier_id,
    )
    if exposure_successor.events[-1].digest != archived.exposure_event_digest:
        raise AtomicSmokePrecommitError(
            "external exposure event differs from precommit"
        )

    _verify_persisted_receipt(
        archived.exposure_persistence_receipt,
        exposure_successor,
        exposure_store_dir,
    )
    authenticated = _authenticate_full_manifest(corpus, full_corpus_manifest)
    frozen = _freeze_selected_task_binding(
        corpus, authenticated, selection.selected_task_id
    )
    fresh_task_manifest = frozen.task.build_manifest()
    if fresh_task_manifest != frozen.trusted_manifest:
        raise AtomicSmokePrecommitError(
            "cold fresh TaskManifest differs from trusted full-manifest entry"
        )
    episode_corpus = ShapeBongardCorpus(
        frozen.corpus_root,
        (frozen.task,),
        layout=corpus.layout,
        split=frozen.split,
    )
    development_manifest = _development_manifest(
        episode_corpus, fresh_task_manifest
    )
    episode_plan = benchmark.prepare_episode(
        episode_corpus,
        selection.selected_task_id,
        seed=f"atomic-smoke-private-episode:{episode_seed}",
        corpus_manifest=development_manifest,
        verifier_id=verifier_id,
        label_seal_nonce=label_seal_nonce,
    )
    replayed = AtomicSmokePrecommit.create(
        selection=selection,
        exposure_successor=exposure_successor,
        exposure_persistence_receipt=archived.exposure_persistence_receipt,
        source_dependency_digest=source_dependency,
        development_manifest=development_manifest,
        episode_plan=episode_plan,
    )
    if replayed.to_data() != archived.to_data():
        raise AtomicSmokePrecommitError(
            "atomic smoke precommit differs from independent cold replay"
        )
    return replayed


__all__ = [
    "ATOMIC_SMOKE_DEVELOPMENT_MANIFEST_SCHEMA",
    "ATOMIC_SMOKE_PRECOMMIT_SCHEMA",
    "ATOMIC_SMOKE_PERSISTENCE_SCHEMA",
    "ATOMIC_SMOKE_SAMPLE_SIZE",
    "ATOMIC_SMOKE_SELECTION_POLICY",
    "ATOMIC_SMOKE_SELECTION_SCHEMA",
    "OFFICIAL_A3_SUCCESSOR_LEDGER_DIGEST",
    "OFFICIAL_A3_SUCCESSOR_UNIVERSE_COUNT",
    "OFFICIAL_A3_SUCCESSOR_UNIVERSE_DIGEST",
    "OFFICIAL_BLOCKED_MORPHOLOGY_POLICY_DIGEST",
    "OFFICIAL_CORPUS_MANIFEST_DIGEST",
    "OFFICIAL_HISTORICAL_SEED_DIGEST",
    "OFFICIAL_RELEASE_DESCRIPTOR_DIGEST",
    "OFFICIAL_RESOLVER_POLICY_DIGEST",
    "OFFICIAL_SPLIT_SOURCE_DIGEST",
    "AtomicSmokePrecommit",
    "AtomicSmokePrecommitError",
    "AtomicSmokeSelection",
    "ExposurePersistenceReceipt",
    "atomic_smoke_protocol_digest",
    "cold_decode_and_replay_atomic_smoke_precommit",
    "prepare_atomic_smoke_precommit",
    "replay_atomic_smoke_selection",
    "select_atomic_smoke_task",
]

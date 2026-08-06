"""Conservative reconstruction of the Bongard historical-exposure seed.

This module keeps three different namespaces separate:

* official task identifiers, which are accepted only from exact repository
  evidence;
* semantic generator families (Basic shapes and Abstract attributes), which
  can be reconstructed from old symbolic and visual runs; and
* source-panel identifiers, for which the old artifacts contain no complete
  audit trail.

In particular, a freshly generated identifier such as ``bd_open_s5_0279`` is
not promoted to an official ShapeBongard_V2 task identifier.  Likewise, the
absence of a local Freeform run is not interpreted as certified non-exposure.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import itertools
import json
from pathlib import Path, PurePosixPath
import subprocess
from typing import Any, Callable, Iterable, Mapping, Sequence


SCHEMA = "gkm.bongard-historical-exposure.v1"
DEFAULT_SEED_PATH = Path(__file__).with_name("data") / "historical_exposure_v1.json"
BASIC_PARTITION_NAMESPACE = "bongard-unused-v1/basic"
ABSTRACT_PAIR_PARTITION_NAMESPACE = "bongard-unused-v1/abstract-pair"
SPLIT_DIGEST_FORMAT = "sha256 of UTF-8 identifiers, one per line, including final newline"
LEGACY_SNAPSHOT_TAG = "pre-bongard-complete-rewrite-20260805"
LEGACY_SNAPSHOT_TAG_OBJECT = "7f7de0f7c335e069197d566cc13efc098a5ceb6d"
LEGACY_SNAPSHOT_COMMIT = "1a71bb1560d41e4a908003a4efd6442e255602fc"
LOGO_FULL_RESULTS_COMMIT = "c250ab87367dccbdfc54e1147f7a3c547c8820a9"

_SNAPSHOT_LEGACY_PATHS = frozenset(
    {
        "bongard/bongard_logo_report.md",
        "bongard/crack_lab/semantic_grounded_runs/codex_eod_20260805_v1/campaign.json",
        "bongard/crack_lab/semantic_grounded_runs/codex_blind_bird6_20260905_v1/campaign.json",
        "bongard/crack_lab/semantic_hybrid_runs/codex_bird6_latent_20260905_v1/campaign.json",
        "bongard/run_bongard_logo_adapter.py",
    }
)
_SPECIAL_LEGACY_COMMITS = {
    "bongard/crack_lab/agent_solutions/logo_full_predicates/results.json": (
        LOGO_FULL_RESULTS_COMMIT
    )
}

EXPECTED_SHAPE_COUNT = 627
EXPECTED_ATTRIBUTE_COUNT = 26
EXPECTED_ADMISSIBLE_PAIR_COUNT = 194
EXPECTED_EXPOSED_BASIC_COUNT = 178
EXPECTED_EXPOSED_PAIR_COUNT = 67
EXPECTED_PARTITION_COUNTS = {"drill": 300, "dev": 75, "sealed": 74}
EXPECTED_ABSTRACT_PAIR_PARTITION_COUNTS = {
    "drill": 85,
    "dev": 21,
    "sealed": 21,
}


class HistoricalExposureError(RuntimeError):
    """The historical seed or the evidence used to reconstruct it is unsafe."""


class _RepositoryEvidenceReader:
    """Read current evidence, with a hash-pinned Git fallback for deleted legacy files."""

    def __init__(
        self,
        repo_root: Path,
        *,
        expected_digests: Mapping[str, str] | None = None,
        snapshot_tag: str = LEGACY_SNAPSHOT_TAG,
        snapshot_tag_object: str = LEGACY_SNAPSHOT_TAG_OBJECT,
        snapshot_commit: str = LEGACY_SNAPSHOT_COMMIT,
        snapshot_paths: Iterable[str] = _SNAPSHOT_LEGACY_PATHS,
        special_commits: Mapping[str, str] = _SPECIAL_LEGACY_COMMITS,
        prefer_pinned_legacy: bool = False,
        runner: Callable[..., subprocess.CompletedProcess[bytes]] = subprocess.run,
    ) -> None:
        self.repo_root = repo_root
        self.expected_digests = dict(expected_digests or {})
        self.snapshot_tag = snapshot_tag
        self.snapshot_tag_object = snapshot_tag_object
        self.snapshot_commit = snapshot_commit
        self.snapshot_paths = frozenset(snapshot_paths)
        self.special_commits = dict(special_commits)
        self.prefer_pinned_legacy = prefer_pinned_legacy
        self.runner = runner
        self._cache: dict[str, bytes] = {}
        self._snapshot_verified = False
        self._verified_commits: set[str] = set()

    @staticmethod
    def _validate_relative_path(relative_path: str) -> None:
        path = PurePosixPath(relative_path)
        if (
            not relative_path
            or path.is_absolute()
            or ".." in path.parts
            or str(path) != relative_path
            or "\x00" in relative_path
            or "\n" in relative_path
            or "\r" in relative_path
        ):
            raise HistoricalExposureError(f"unsafe evidence path: {relative_path!r}")

    def _git(self, *arguments: str) -> bytes:
        try:
            result = self.runner(
                ["git", *arguments],
                cwd=str(self.repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=15,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise HistoricalExposureError(f"cannot read pinned Git evidence: {exc}") from exc
        if result.returncode != 0:
            detail = result.stderr.decode("utf-8", errors="replace").strip()
            raise HistoricalExposureError(
                f"pinned Git evidence command failed ({' '.join(arguments)}): {detail}"
            )
        return bytes(result.stdout)

    def _verify_snapshot(self) -> None:
        if self._snapshot_verified:
            return
        tag_object = self._git(
            "rev-parse", "--verify", f"refs/tags/{self.snapshot_tag}"
        ).decode("ascii").strip()
        if tag_object != self.snapshot_tag_object:
            raise HistoricalExposureError(
                f"legacy snapshot tag object is {tag_object!r}, expected {self.snapshot_tag_object!r}"
            )
        object_type = self._git("cat-file", "-t", tag_object).decode("ascii").strip()
        if object_type != "tag":
            raise HistoricalExposureError("legacy snapshot must be an annotated Git tag")
        commit = self._git(
            "rev-parse", "--verify", f"{self.snapshot_tag}^{{commit}}"
        ).decode("ascii").strip()
        if commit != self.snapshot_commit:
            raise HistoricalExposureError(
                f"legacy snapshot resolves to {commit!r}, expected {self.snapshot_commit!r}"
            )
        self._snapshot_verified = True

    def _verify_commit(self, commit: str) -> None:
        if commit in self._verified_commits:
            return
        object_type = self._git("cat-file", "-t", commit).decode("ascii").strip()
        if object_type != "commit":
            raise HistoricalExposureError(f"legacy fallback {commit!r} is not a commit")
        resolved = self._git("rev-parse", "--verify", f"{commit}^{{commit}}").decode(
            "ascii"
        ).strip()
        if resolved != commit:
            raise HistoricalExposureError(
                f"legacy commit resolves to {resolved!r}, expected {commit!r}"
            )
        self._verified_commits.add(commit)

    def _fallback_bytes(self, relative_path: str) -> bytes:
        if relative_path in self.snapshot_paths:
            self._verify_snapshot()
            commit = self.snapshot_commit
        elif relative_path in self.special_commits:
            commit = self.special_commits[relative_path]
            self._verify_commit(commit)
        else:
            raise HistoricalExposureError(
                f"evidence is missing and has no pinned fallback: {relative_path}"
            )
        return self._git("show", f"{commit}:{relative_path}")

    def read_bytes(self, relative_path: str) -> bytes:
        self._validate_relative_path(relative_path)
        if relative_path in self._cache:
            return self._cache[relative_path]
        local_path = self.repo_root / relative_path
        has_pinned_fallback = (
            relative_path in self.snapshot_paths or relative_path in self.special_commits
        )
        if self.prefer_pinned_legacy and has_pinned_fallback:
            payload = self._fallback_bytes(relative_path)
        elif local_path.is_file():
            try:
                payload = local_path.read_bytes()
            except OSError as exc:
                raise HistoricalExposureError(
                    f"cannot read historical evidence {local_path}: {exc}"
                ) from exc
        elif local_path.exists():
            raise HistoricalExposureError(f"historical evidence is not a file: {local_path}")
        else:
            payload = self._fallback_bytes(relative_path)

        actual = "sha256:" + hashlib.sha256(payload).hexdigest()
        expected = self.expected_digests.get(relative_path)
        if expected is not None and actual != expected:
            raise HistoricalExposureError(
                f"historical evidence hash mismatch for {relative_path}: "
                f"{actual} != {expected}"
            )
        self._cache[relative_path] = payload
        return payload

    def read_text(self, relative_path: str) -> str:
        try:
            return self.read_bytes(relative_path).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise HistoricalExposureError(
                f"historical evidence is not UTF-8: {relative_path}"
            ) from exc

    def address(self, relative_path: str) -> str:
        return "sha256:" + hashlib.sha256(self.read_bytes(relative_path)).hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _address(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _line_digest(values: Iterable[str]) -> str:
    items = tuple(values)
    if not all(isinstance(item, str) and item for item in items):
        raise HistoricalExposureError("digest members must be non-empty strings")
    payload = "".join(f"{item}\n" for item in items).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _unique_strings(value: Any, *, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise HistoricalExposureError(f"{label} must be a JSON list of non-empty strings")
    result = tuple(value)
    if len(result) != len(set(result)):
        raise HistoricalExposureError(f"{label} contains duplicate identifiers")
    return result


def _unique_pairs(value: Any, *, label: str) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, list):
        raise HistoricalExposureError(f"{label} must be a JSON list")
    result: list[tuple[str, str]] = []
    for item in value:
        if (
            not isinstance(item, list)
            or len(item) != 2
            or not all(isinstance(member, str) and member for member in item)
            or item[0] == item[1]
        ):
            raise HistoricalExposureError(f"{label} contains an invalid pair: {item!r}")
        result.append((item[0], item[1]))
    if len(result) != len(set(result)):
        raise HistoricalExposureError(f"{label} contains duplicate pairs")
    return tuple(result)


def _require_object(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise HistoricalExposureError(f"{label} must be a JSON object")
    return value


def _require_fields(raw: Mapping[str, Any], fields: set[str], *, label: str) -> None:
    actual = set(raw)
    if actual != fields:
        raise HistoricalExposureError(
            f"{label} fields differ from schema: missing={sorted(fields - actual)}, "
            f"extra={sorted(actual - fields)}"
        )


@dataclass(frozen=True)
class BasicFamilyPartition:
    """The fully resolved family-clean Basic drill/dev/sealed partition."""

    namespace: str
    drill: tuple[str, ...]
    dev: tuple[str, ...]
    sealed: tuple[str, ...]
    drill_digest: str
    dev_digest: str
    sealed_digest: str

    @property
    def eligible(self) -> tuple[str, ...]:
        return self.drill + self.dev + self.sealed

    @property
    def counts(self) -> Mapping[str, int]:
        return {
            "drill": len(self.drill),
            "dev": len(self.dev),
            "sealed": len(self.sealed),
        }


def _pair_identifier(pair: tuple[str, str]) -> str:
    """Return the unambiguous line representation used for pair ranking/digests."""

    return "\t".join(pair)


@dataclass(frozen=True)
class AbstractPairPartition:
    """A frozen partition whose indivisible unit is one ordered Abstract pair."""

    namespace: str
    drill: tuple[tuple[str, str], ...]
    dev: tuple[tuple[str, str], ...]
    sealed: tuple[tuple[str, str], ...]
    drill_digest: str
    dev_digest: str
    sealed_digest: str

    def __post_init__(self) -> None:
        if self.namespace != ABSTRACT_PAIR_PARTITION_NAMESPACE:
            raise HistoricalExposureError("Abstract pair partition uses an unsupported namespace")
        groups = {"drill": self.drill, "dev": self.dev, "sealed": self.sealed}
        if {name: len(values) for name, values in groups.items()} != (
            EXPECTED_ABSTRACT_PAIR_PARTITION_COUNTS
        ):
            raise HistoricalExposureError(
                "Abstract pair partition does not have the frozen 85/21/21 counts"
            )
        for name, values in groups.items():
            if len(values) != len(set(values)):
                raise HistoricalExposureError(
                    f"Abstract pair partition {name} contains duplicate pairs"
                )
            for pair in values:
                if (
                    not isinstance(pair, tuple)
                    or len(pair) != 2
                    or not all(isinstance(item, str) and item for item in pair)
                    or pair[0] == pair[1]
                ):
                    raise HistoricalExposureError(
                        f"Abstract pair partition contains an invalid pair: {pair!r}"
                    )
        if (
            set(self.drill) & set(self.dev)
            or set(self.drill) & set(self.sealed)
            or set(self.dev) & set(self.sealed)
        ):
            raise HistoricalExposureError("Abstract pair partition members overlap")
        expected_digests = {
            name: _line_digest(_pair_identifier(pair) for pair in values)
            for name, values in groups.items()
        }
        actual_digests = {
            "drill": self.drill_digest,
            "dev": self.dev_digest,
            "sealed": self.sealed_digest,
        }
        if actual_digests != expected_digests:
            raise HistoricalExposureError("Abstract pair partition digest mismatch")

    @property
    def eligible(self) -> tuple[tuple[str, str], ...]:
        return self.drill + self.dev + self.sealed

    @property
    def counts(self) -> Mapping[str, int]:
        return {
            "drill": len(self.drill),
            "dev": len(self.dev),
            "sealed": len(self.sealed),
        }


@dataclass(frozen=True)
class HistoricalExposureSeed:
    """Validated immutable view of ``historical_exposure_v1.json``."""

    exact_official_task_ids: tuple[str, ...]
    exact_official_panel_ids: tuple[str, ...]
    basic_shape_families: tuple[str, ...]
    visual_basic_shape_families: tuple[str, ...]
    abstract_attributes: tuple[str, ...]
    visual_abstract_attributes: tuple[str, ...]
    abstract_pairs: tuple[tuple[str, str], ...]
    freeform_status: str
    freeform_exact_task_ids: tuple[str, ...]
    partition: BasicFamilyPartition
    abstract_partition: AbstractPairPartition
    admissible_abstract_pairs: tuple[tuple[str, str], ...]
    evidence_files: tuple[tuple[str, str], ...]
    seed_digest: str
    raw: Mapping[str, Any]

    @property
    def unused_basic_shape_families(self) -> tuple[str, ...]:
        return self.partition.eligible

    @property
    def unused_abstract_pairs(self) -> tuple[tuple[str, str], ...]:
        return self.abstract_partition.eligible


def _read_tsv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if reader.fieldnames is None:
                raise HistoricalExposureError(f"TSV has no header: {path}")
            return list(reader.fieldnames), list(reader)
    except OSError as exc:
        raise HistoricalExposureError(f"cannot read generator metadata {path}: {exc}") from exc


def _shape_inventory(dataset_root: Path) -> tuple[str, ...]:
    path = dataset_root / "data" / "human_designed_shapes.tsv"
    fields, rows = _read_tsv(path)
    if "shape function name" not in fields:
        raise HistoricalExposureError(f"missing shape function name column in {path}")
    shapes = tuple(row.get("shape function name", "") for row in rows)
    if len(shapes) != EXPECTED_SHAPE_COUNT:
        raise HistoricalExposureError(
            f"shape inventory has {len(shapes)} rows, expected {EXPECTED_SHAPE_COUNT}"
        )
    if not all(shapes) or len(shapes) != len(set(shapes)):
        raise HistoricalExposureError("shape inventory contains an empty or duplicate family")
    return shapes


def _attribute_candidates(
    dataset_root: Path,
) -> tuple[tuple[str, ...], Mapping[str, tuple[frozenset[str], frozenset[str]]]]:
    """Reproduce ``get_attribute_sampling_candidates`` without importing pandas."""

    path = dataset_root / "data" / "human_designed_shapes_attributes.tsv"
    fields, rows = _read_tsv(path)
    required = {"shape function name", "symmetric", "self_transposed"}
    if not required <= set(fields) or len(fields) < 4:
        raise HistoricalExposureError(f"attribute TSV is missing required columns: {path}")

    base_attributes = fields[3:]
    if "symmetric_transposed" in base_attributes:
        raise HistoricalExposureError("derived symmetric_transposed column is already present")
    attributes = tuple(base_attributes + ["symmetric_transposed"])
    if len(attributes) != EXPECTED_ATTRIBUTE_COUNT or len(attributes) != len(set(attributes)):
        raise HistoricalExposureError(
            f"attribute inventory has {len(attributes)} unique columns, "
            f"expected {EXPECTED_ATTRIBUTE_COUNT}"
        )

    values: dict[str, dict[str, int]] = {attribute: {} for attribute in attributes}
    seen_shapes: set[str] = set()
    for row_index, row in enumerate(rows, start=2):
        shape = row.get("shape function name", "")
        if not shape or shape in seen_shapes:
            raise HistoricalExposureError(f"empty or duplicate shape at {path}:{row_index}")
        seen_shapes.add(shape)
        for attribute in base_attributes:
            try:
                values[attribute][shape] = int(row[attribute])
            except (KeyError, TypeError, ValueError) as exc:
                raise HistoricalExposureError(
                    f"invalid {attribute!r} value at {path}:{row_index}"
                ) from exc
        symmetric = values["symmetric"][shape]
        transposed = values["self_transposed"][shape]
        if symmetric == 1 and transposed == 1:
            derived = 1
        elif symmetric == -1 or transposed == -1:
            derived = -1
        elif symmetric == 0 or transposed == 0:
            derived = 0
        else:
            derived = -1
        values["symmetric_transposed"][shape] = derived

    candidates: dict[str, tuple[frozenset[str], frozenset[str]]] = {}
    for attribute in attributes:
        positive = frozenset(shape for shape, value in values[attribute].items() if value == 1)
        negative = frozenset(shape for shape, value in values[attribute].items() if value == 0)
        if not positive or not negative:
            raise HistoricalExposureError(
                f"attribute {attribute!r} lacks a positive or negative sampling bucket"
            )
        candidates[attribute] = positive, negative
    return attributes, candidates


def _pair_has_capacity(
    candidates: Mapping[str, tuple[frozenset[str], frozenset[str]]],
    pair: tuple[str, str],
    total_examples: int,
) -> bool:
    pos0, neg0 = candidates[pair[0]]
    pos1, neg1 = candidates[pair[1]]
    required_negative_bucket = max(10, (total_examples + 1) // 2)
    return (
        len(pos0 & pos1) >= max(total_examples, 10)
        and len(pos0 & neg1) >= required_negative_bucket
        and len(neg0 & pos1) >= required_negative_bucket
    )


def _viable_pairs(
    attributes: Sequence[str],
    candidates: Mapping[str, tuple[frozenset[str], frozenset[str]]],
    *,
    total_examples: int,
) -> tuple[tuple[str, str], ...]:
    return tuple(
        pair
        for pair in itertools.combinations(attributes, 2)
        if _pair_has_capacity(candidates, pair, total_examples)
    )


def _load_json_object(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HistoricalExposureError(f"cannot read {label} {path}: {exc}") from exc
    return _require_object(raw, label=label)


def _load_evidence_json(
    reader: _RepositoryEvidenceReader, relative_path: str, *, label: str
) -> Mapping[str, Any]:
    try:
        raw = json.loads(reader.read_text(relative_path))
    except json.JSONDecodeError as exc:
        raise HistoricalExposureError(
            f"cannot decode {label} {relative_path}: {exc}"
        ) from exc
    return _require_object(raw, label=label)


def _stage_one_concepts(
    reader: _RepositoryEvidenceReader,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    relative_path = "bongard/crack_lab/agent_solutions/logo_full_predicates/results.json"
    raw = _load_evidence_json(reader, relative_path, label="stage-1 results")
    expected_keys = [f"problem_{index:02d}" for index in range(80)]
    if list(raw) != expected_keys:
        raise HistoricalExposureError("stage-1 results are not the exact ordered problem_00..problem_79 corpus")

    basic: list[str] = []
    abstract: list[str] = []
    for opaque_id in expected_keys:
        record = _require_object(raw[opaque_id], label=f"stage-1 record {opaque_id}")
        category = record.get("category")
        concept = record.get("concept")
        problem_id = record.get("problem_id")
        if category not in {"basic", "abstract"}:
            raise HistoricalExposureError(f"stage-1 {opaque_id} has invalid category {category!r}")
        if not isinstance(concept, str) or not concept or not isinstance(problem_id, str):
            raise HistoricalExposureError(f"stage-1 {opaque_id} lacks an exact concept/problem id")
        expected_prefix = "bd_" if category == "basic" else "hd_"
        if not problem_id.startswith(expected_prefix):
            raise HistoricalExposureError(f"stage-1 {opaque_id} has inconsistent generated id")
        (basic if category == "basic" else abstract).append(concept)

    if len(basic) != 64 or len(abstract) != 16:
        raise HistoricalExposureError(
            f"stage-1 category counts are basic={len(basic)}, abstract={len(abstract)}; "
            "expected 64 and 16"
        )
    if len(set(basic)) != len(basic) or len(set(abstract)) != len(abstract):
        raise HistoricalExposureError("stage-1 contains duplicate semantic concepts")
    return tuple(basic), tuple(abstract)


def _grounded_concept(reader: _RepositoryEvidenceReader, relative_path: str) -> str:
    raw = _load_evidence_json(reader, relative_path, label="grounded campaign")
    records = raw.get("records")
    if not isinstance(records, list) or len(records) != 1:
        raise HistoricalExposureError(f"{relative_path} must contain exactly one campaign record")
    record = _require_object(records[0], label=f"{relative_path} record")
    metadata = _require_object(record.get("generator_metadata"), label="generator_metadata")
    if metadata.get("category") != "basic":
        raise HistoricalExposureError(f"{relative_path} is not a Basic exposure")
    concept = metadata.get("concept")
    if not isinstance(concept, str) or not concept:
        raise HistoricalExposureError(f"{relative_path} has no exact Basic concept")
    return concept


def _hybrid_concept(reader: _RepositoryEvidenceReader, relative_path: str) -> str:
    raw = _load_evidence_json(reader, relative_path, label="hybrid campaign")
    records = raw.get("records")
    if not isinstance(records, list) or len(records) != 1:
        raise HistoricalExposureError(f"{relative_path} must contain exactly one campaign record")
    record = _require_object(records[0], label=f"{relative_path} record")
    split = _require_object(record.get("program_split"), label="program_split")
    if split.get("category") != "basic":
        raise HistoricalExposureError(f"{relative_path} is not a Basic exposure")
    concept = split.get("concept")
    if not isinstance(concept, str) or not concept:
        raise HistoricalExposureError(f"{relative_path} has no exact Basic concept")
    return concept


def _ordered_union(*groups: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for group in groups:
        for item in group:
            if item not in seen:
                seen.add(item)
                result.append(item)
    return tuple(result)


def _partition_basic(
    shape_inventory: Sequence[str], exposed: Iterable[str]
) -> BasicFamilyPartition:
    exposed_set = set(exposed)
    unknown = exposed_set - set(shape_inventory)
    if unknown:
        raise HistoricalExposureError(f"exposed Basic families absent from inventory: {sorted(unknown)}")
    eligible = set(shape_inventory) - exposed_set

    def rank(shape: str) -> tuple[str, str]:
        value = f"{BASIC_PARTITION_NAMESPACE}/{shape}".encode("utf-8")
        return hashlib.sha256(value).hexdigest(), shape

    ordered = tuple(shape for _digest, shape in sorted(rank(shape) for shape in eligible))
    drill = ordered[: EXPECTED_PARTITION_COUNTS["drill"]]
    dev_end = len(drill) + EXPECTED_PARTITION_COUNTS["dev"]
    dev = ordered[len(drill) : dev_end]
    sealed = ordered[dev_end:]
    partition = BasicFamilyPartition(
        namespace=BASIC_PARTITION_NAMESPACE,
        drill=drill,
        dev=dev,
        sealed=sealed,
        drill_digest=_line_digest(drill),
        dev_digest=_line_digest(dev),
        sealed_digest=_line_digest(sealed),
    )
    if dict(partition.counts) != EXPECTED_PARTITION_COUNTS:
        raise HistoricalExposureError(
            f"family-clean Basic partition has counts {dict(partition.counts)}, "
            f"expected {EXPECTED_PARTITION_COUNTS}"
        )
    return partition


def _partition_abstract_pairs(
    admissible_pairs: Sequence[tuple[str, str]],
    exposed_pairs: Iterable[tuple[str, str]],
) -> AbstractPairPartition:
    """Hash-rank unused ordered pairs without ever splitting their instances."""

    admissible = tuple(admissible_pairs)
    if len(admissible) != len(set(admissible)):
        raise HistoricalExposureError("admissible Abstract pair inventory contains duplicates")
    exposed = set(exposed_pairs)
    unknown = exposed - set(admissible)
    if unknown:
        raise HistoricalExposureError(
            f"exposed Abstract pairs absent from inventory: {sorted(unknown)}"
        )
    eligible = set(admissible) - exposed

    def rank(pair: tuple[str, str]) -> tuple[str, tuple[str, str]]:
        value = (
            f"{ABSTRACT_PAIR_PARTITION_NAMESPACE}/{_pair_identifier(pair)}"
        ).encode("utf-8")
        return hashlib.sha256(value).hexdigest(), pair

    ordered = tuple(pair for _digest, pair in sorted(rank(pair) for pair in eligible))
    drill = ordered[: EXPECTED_ABSTRACT_PAIR_PARTITION_COUNTS["drill"]]
    dev_end = len(drill) + EXPECTED_ABSTRACT_PAIR_PARTITION_COUNTS["dev"]
    dev = ordered[len(drill) : dev_end]
    sealed = ordered[dev_end:]
    return AbstractPairPartition(
        namespace=ABSTRACT_PAIR_PARTITION_NAMESPACE,
        drill=drill,
        dev=dev,
        sealed=sealed,
        drill_digest=_line_digest(_pair_identifier(pair) for pair in drill),
        dev_digest=_line_digest(_pair_identifier(pair) for pair in dev),
        sealed_digest=_line_digest(_pair_identifier(pair) for pair in sealed),
    )


_EVIDENCE_SPECS = (
    ("bongard/bongard_logo_report.md", "symbolic Basic/Abstract run report", None),
    (
        "bongard/crack_lab/agent_solutions/logo_full_predicates/results.json",
        "stage-1 visual results",
        "c250ab87",
    ),
    (
        "bongard/crack_lab/semantic_grounded_runs/codex_eod_20260805_v1/campaign.json",
        "mismatch_sector_rec2 visual campaign",
        None,
    ),
    (
        "bongard/crack_lab/semantic_grounded_runs/codex_blind_bird6_20260905_v1/campaign.json",
        "bird6 blind visual campaign",
        None,
    ),
    (
        "bongard/crack_lab/semantic_hybrid_runs/codex_bird6_latent_20260905_v1/campaign.json",
        "bird6 hybrid visual campaign corroboration",
        None,
    ),
    ("downloads/Bongard-LOGO/README.md", "published exact demo identities", None),
    (
        "downloads/Bongard-LOGO/data/human_designed_shapes.tsv",
        "ordered Basic generator inventory",
        None,
    ),
    (
        "downloads/Bongard-LOGO/data/human_designed_shapes_attributes.tsv",
        "ordered Abstract generator inventory",
        None,
    ),
    ("downloads/Bongard-LOGO/bongard/util_funcs.py", "upstream metadata semantics", None),
    ("bongard/run_bongard_logo_adapter.py", "symbolic concept/capacity semantics", None),
)


def _build_seed_body(
    repo_root: Path,
    dataset_root: Path,
    reader: _RepositoryEvidenceReader,
) -> tuple[
    dict[str, Any],
    BasicFamilyPartition,
    AbstractPairPartition,
    tuple[tuple[str, str], ...],
]:
    shapes = _shape_inventory(dataset_root)
    attributes, candidates = _attribute_candidates(dataset_root)

    report_path = "bongard/bongard_logo_report.md"
    report = reader.read_text(report_path)
    for required_text in (
        "--source basic --feature-set action --limit 120",
        "For one-attribute Abstract concepts",
        "On the first 80 viable concepts",
        "support=20/5/5",
        "support=10/3/3",
    ):
        if required_text not in report:
            raise HistoricalExposureError(
                f"symbolic report no longer proves required claim: {required_text!r}"
            )

    symbolic_basic = tuple(shapes[:120])
    visual_stage_basic, visual_abstract = _stage_one_concepts(reader)
    mismatch_path = (
        "bongard/crack_lab/semantic_grounded_runs/"
        "codex_eod_20260805_v1/campaign.json"
    )
    bird_grounded_path = (
        "bongard/crack_lab/semantic_grounded_runs/"
        "codex_blind_bird6_20260905_v1/campaign.json"
    )
    bird_hybrid_path = (
        "bongard/crack_lab/semantic_hybrid_runs/"
        "codex_bird6_latent_20260905_v1/campaign.json"
    )
    mismatch = _grounded_concept(reader, mismatch_path)
    bird_grounded = _grounded_concept(reader, bird_grounded_path)
    bird_hybrid = _hybrid_concept(reader, bird_hybrid_path)
    if mismatch != "mismatch_sector_rec2":
        raise HistoricalExposureError(
            f"20260805 grounded campaign resolves to {mismatch!r}, expected mismatch_sector_rec2"
        )
    if bird_grounded != "bird6" or bird_hybrid != bird_grounded:
        raise HistoricalExposureError("grounded/hybrid bird6 evidence disagrees")

    visual_basic = _ordered_union(visual_stage_basic, (mismatch, bird_grounded))
    exposed_basic = _ordered_union(symbolic_basic, visual_basic)
    if len(visual_basic) != 66 or len(set(symbolic_basic) & set(visual_basic)) != 8:
        raise HistoricalExposureError(
            "visual Basic reconstruction must contain 66 families with eight symbolic overlaps"
        )
    if len(exposed_basic) != EXPECTED_EXPOSED_BASIC_COUNT:
        raise HistoricalExposureError(
            f"Basic exposure union has {len(exposed_basic)}, expected {EXPECTED_EXPOSED_BASIC_COUNT}"
        )

    admissible_pairs = _viable_pairs(attributes, candidates, total_examples=7)
    if len(admissible_pairs) != EXPECTED_ADMISSIBLE_PAIR_COUNT:
        raise HistoricalExposureError(
            f"admissible Abstract pair inventory has {len(admissible_pairs)}, "
            f"expected {EXPECTED_ADMISSIBLE_PAIR_COUNT}"
        )
    pair_run_16 = _viable_pairs(attributes, candidates, total_examples=16)[:54]
    pair_run_30 = _viable_pairs(attributes, candidates, total_examples=30)[:54]
    exposed_pairs = tuple(
        dict.fromkeys(pair_run_16 + pair_run_30)
    )
    if len(exposed_pairs) != EXPECTED_EXPOSED_PAIR_COUNT:
        raise HistoricalExposureError(
            f"Abstract pair exposure union has {len(exposed_pairs)}, "
            f"expected {EXPECTED_EXPOSED_PAIR_COUNT}"
        )
    unknown_visual_attributes = set(visual_abstract) - set(attributes)
    if len(visual_abstract) != 16 or unknown_visual_attributes:
        raise HistoricalExposureError(
            f"invalid visual Abstract inventory: unknown={sorted(unknown_visual_attributes)}"
        )

    partition = _partition_basic(shapes, exposed_basic)
    abstract_partition = _partition_abstract_pairs(admissible_pairs, exposed_pairs)
    demo_ids = (
        ("ff_nact6_0292", "ff"),
        ("bd_isosceles_trapezoid-no_obtuse_angle_six_lines2_0000", "bd"),
        ("hd_convex_0004", "hd"),
    )
    upstream_readme = reader.read_text("downloads/Bongard-LOGO/README.md")
    missing_demo_ids = [task_id for task_id, _family in demo_ids if task_id not in upstream_readme]
    if missing_demo_ids:
        raise HistoricalExposureError(
            f"upstream README no longer evidences demo ids: {missing_demo_ids}"
        )

    evidence = []
    for relative_path, purpose, commit in _EVIDENCE_SPECS:
        record: dict[str, Any] = {
            "path": relative_path,
            "purpose": purpose,
            "sha256": reader.address(relative_path),
        }
        if commit is not None:
            record["historical_commit"] = commit
        evidence.append(record)

    body: dict[str, Any] = {
        "schema": SCHEMA,
        "qualification": {
            "exact_task_ids_are_not_panel_ids": True,
            "fresh_generator_ids_are_not_official_task_ids": True,
            "official_panel_bytes_evidenced": False,
            "semantic_families_are_not_projected_to_official_task_ids": True,
        },
        "evidence_files": evidence,
        "exact_official_exposure": {
            "task_ids": [
                {
                    "task_id": task_id,
                    "family": family,
                    "basis": "public-demo identity; conservatively exclude at exact-task level",
                    "evidence_path": "downloads/Bongard-LOGO/README.md",
                }
                for task_id, family in demo_ids
            ],
            "panel_ids": [],
        },
        "generator_inventory": {
            "basic_shape_count": len(shapes),
            "basic_shape_order_digest": _line_digest(shapes),
            "abstract_attribute_count": len(attributes),
            "abstract_attribute_order_digest": _line_digest(attributes),
            "admissible_abstract_pair_count": len(admissible_pairs),
            "admissible_abstract_pair_digest": _line_digest(
                "\t".join(pair) for pair in admissible_pairs
            ),
            "admissible_abstract_pairs": [list(pair) for pair in admissible_pairs],
        },
        "semantic_exposure": {
            "basic": {
                "shape_families": list(exposed_basic),
                "count": len(exposed_basic),
                "digest": _line_digest(exposed_basic),
                "symbolic_generator_prefix_count": len(symbolic_basic),
                "visual_shape_families": list(visual_basic),
                "visual_count": len(visual_basic),
                "symbolic_visual_overlap_count": len(set(symbolic_basic) & set(visual_basic)),
            },
            "abstract": {
                "attributes": list(attributes),
                "attribute_count": len(attributes),
                "attribute_digest": _line_digest(attributes),
                "visual_attributes": list(visual_abstract),
                "visual_attribute_count": len(visual_abstract),
                "pairs": [list(pair) for pair in exposed_pairs],
                "pair_count": len(exposed_pairs),
                "pair_digest": _line_digest("\t".join(pair) for pair in exposed_pairs),
                "pair_run_definition": {
                    "limit": 80,
                    "single_attribute_prefix": 26,
                    "pair_slots_per_run": 54,
                    "total_examples": [16, 30],
                },
            },
            "freeform": {
                "status": "indeterminate",
                "exact_task_ids": ["ff_nact6_0292"],
                "semantic_family_partition_available": False,
                "reason": (
                    "No local Freeform model-run artifact was found. Public-gallery and "
                    "foundation-model exposure cannot be excluded, so absence is not certified."
                ),
            },
        },
        "unused": {
            "basic_shape_count": len(partition.eligible),
            "abstract_pair_count": len(admissible_pairs) - len(exposed_pairs),
            "freeform_count": None,
        },
        "basic_partition": {
            "namespace": partition.namespace,
            "rank": "ascending sha256(namespace + '/' + shape_family), tie-break shape_family",
            "split_policy": "ranked contiguous prefixes: drill, then dev, then sealed",
            "digest_format": SPLIT_DIGEST_FORMAT,
            "counts": dict(partition.counts),
            "digests": {
                "drill": partition.drill_digest,
                "dev": partition.dev_digest,
                "sealed": partition.sealed_digest,
            },
            "first_ids": {
                "drill": list(partition.drill[:10]),
                "dev": list(partition.dev[:10]),
                "sealed": list(partition.sealed[:10]),
            },
            "members": {
                "drill": list(partition.drill),
                "dev": list(partition.dev),
                "sealed": list(partition.sealed),
            },
        },
        "abstract_pair_partition": {
            "namespace": abstract_partition.namespace,
            "rank": (
                "ascending sha256(namespace + '/' + first_attribute + tab + "
                "second_attribute), tie-break ordered pair"
            ),
            "split_policy": "ranked contiguous prefixes: drill, then dev, then sealed",
            "digest_format": SPLIT_DIGEST_FORMAT,
            "counts": dict(abstract_partition.counts),
            "digests": {
                "drill": abstract_partition.drill_digest,
                "dev": abstract_partition.dev_digest,
                "sealed": abstract_partition.sealed_digest,
            },
            "first_ids": {
                "drill": [list(pair) for pair in abstract_partition.drill[:10]],
                "dev": [list(pair) for pair in abstract_partition.dev[:10]],
                "sealed": [list(pair) for pair in abstract_partition.sealed[:10]],
            },
            "members": {
                "drill": [list(pair) for pair in abstract_partition.drill],
                "dev": [list(pair) for pair in abstract_partition.dev],
                "sealed": [list(pair) for pair in abstract_partition.sealed],
            },
            "task_instances_per_pair": 20,
            "task_counts": {
                name: count * 20
                for name, count in abstract_partition.counts.items()
            },
        },
    }
    return body, partition, abstract_partition, admissible_pairs


def _seed_from_raw(
    raw: Mapping[str, Any],
    *,
    partition: BasicFamilyPartition,
    abstract_partition: AbstractPairPartition,
    admissible_pairs: tuple[tuple[str, str], ...],
) -> HistoricalExposureSeed:
    _require_fields(raw, {"seed", "seed_digest"}, label="historical exposure envelope")
    body = _require_object(raw["seed"], label="historical exposure seed")
    if body.get("schema") != SCHEMA:
        raise HistoricalExposureError(f"unsupported historical exposure schema: {body.get('schema')!r}")
    if raw["seed_digest"] != _address(body):
        raise HistoricalExposureError("historical exposure seed digest mismatch")

    exact = _require_object(body.get("exact_official_exposure"), label="exact_official_exposure")
    task_records = exact.get("task_ids")
    if not isinstance(task_records, list):
        raise HistoricalExposureError("exact official task records must be a list")
    exact_tasks: list[str] = []
    for record in task_records:
        item = _require_object(record, label="exact official task record")
        task_id = item.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            raise HistoricalExposureError("exact official task record lacks task_id")
        exact_tasks.append(task_id)
    if len(exact_tasks) != len(set(exact_tasks)):
        raise HistoricalExposureError("exact official task exposure contains duplicates")
    panels = _unique_strings(exact.get("panel_ids"), label="exact official panel_ids")

    semantic = _require_object(body.get("semantic_exposure"), label="semantic_exposure")
    basic = _require_object(semantic.get("basic"), label="semantic_exposure.basic")
    abstract = _require_object(semantic.get("abstract"), label="semantic_exposure.abstract")
    freeform = _require_object(semantic.get("freeform"), label="semantic_exposure.freeform")
    basic_shapes = _unique_strings(basic.get("shape_families"), label="basic.shape_families")
    visual_basic = _unique_strings(
        basic.get("visual_shape_families"), label="basic.visual_shape_families"
    )
    attributes = _unique_strings(abstract.get("attributes"), label="abstract.attributes")
    visual_attributes = _unique_strings(
        abstract.get("visual_attributes"), label="abstract.visual_attributes"
    )
    pairs = _unique_pairs(abstract.get("pairs"), label="abstract.pairs")

    expected_scalars = (
        (basic.get("count"), len(basic_shapes), "basic count"),
        (basic.get("visual_count"), len(visual_basic), "visual Basic count"),
        (abstract.get("attribute_count"), len(attributes), "attribute count"),
        (abstract.get("visual_attribute_count"), len(visual_attributes), "visual attribute count"),
        (abstract.get("pair_count"), len(pairs), "pair count"),
    )
    for stored, actual, label in expected_scalars:
        if stored != actual:
            raise HistoricalExposureError(f"{label} is {stored!r}, expected {actual}")
    if len(basic_shapes) != EXPECTED_EXPOSED_BASIC_COUNT:
        raise HistoricalExposureError("historical Basic exposure must contain 178 families")
    if len(visual_basic) != 66 or len(set(basic_shapes) & set(visual_basic)) != 66:
        raise HistoricalExposureError("historical visual Basic exposure must contain 66 known families")
    if len(attributes) != EXPECTED_ATTRIBUTE_COUNT or len(visual_attributes) != 16:
        raise HistoricalExposureError("historical Abstract attribute counts are not 26/16")
    if len(pairs) != EXPECTED_EXPOSED_PAIR_COUNT:
        raise HistoricalExposureError("historical Abstract pair exposure must contain 67 pairs")
    expected_digests = (
        (basic.get("digest"), _line_digest(basic_shapes), "Basic exposure"),
        (abstract.get("attribute_digest"), _line_digest(attributes), "attribute exposure"),
        (
            abstract.get("pair_digest"),
            _line_digest("\t".join(pair) for pair in pairs),
            "pair exposure",
        ),
    )
    for stored, actual, label in expected_digests:
        if stored != actual:
            raise HistoricalExposureError(f"{label} digest mismatch")

    partition_raw = _require_object(body.get("basic_partition"), label="basic_partition")
    if partition.namespace != BASIC_PARTITION_NAMESPACE:
        raise HistoricalExposureError("Basic partition uses an unsupported namespace")
    if dict(partition.counts) != EXPECTED_PARTITION_COUNTS:
        raise HistoricalExposureError("Basic partition does not have the frozen 300/75/74 counts")
    if len(partition.eligible) != 449 or len(set(partition.eligible)) != 449:
        raise HistoricalExposureError("Basic partition must contain exactly 449 unique families")
    if set(partition.eligible) & set(basic_shapes):
        raise HistoricalExposureError("Basic partition includes a historically exposed family")
    if partition_raw.get("namespace") != partition.namespace:
        raise HistoricalExposureError("Basic partition namespace mismatch")
    if partition_raw.get("counts") != dict(partition.counts):
        raise HistoricalExposureError("Basic partition counts mismatch")
    if partition_raw.get("digests") != {
        "drill": partition.drill_digest,
        "dev": partition.dev_digest,
        "sealed": partition.sealed_digest,
    }:
        raise HistoricalExposureError("Basic partition digest mismatch")
    if partition_raw.get("first_ids") != {
        "drill": list(partition.drill[:10]),
        "dev": list(partition.dev[:10]),
        "sealed": list(partition.sealed[:10]),
    }:
        raise HistoricalExposureError("Basic partition first-id witness mismatch")

    abstract_partition_raw = _require_object(
        body.get("abstract_pair_partition"), label="abstract_pair_partition"
    )
    if abstract_partition_raw.get("namespace") != abstract_partition.namespace:
        raise HistoricalExposureError("Abstract pair partition namespace mismatch")
    if abstract_partition_raw.get("counts") != dict(abstract_partition.counts):
        raise HistoricalExposureError("Abstract pair partition counts mismatch")
    if abstract_partition_raw.get("digests") != {
        "drill": abstract_partition.drill_digest,
        "dev": abstract_partition.dev_digest,
        "sealed": abstract_partition.sealed_digest,
    }:
        raise HistoricalExposureError("Abstract pair partition digest mismatch")
    if abstract_partition_raw.get("first_ids") != {
        "drill": [list(pair) for pair in abstract_partition.drill[:10]],
        "dev": [list(pair) for pair in abstract_partition.dev[:10]],
        "sealed": [list(pair) for pair in abstract_partition.sealed[:10]],
    }:
        raise HistoricalExposureError("Abstract pair partition first-id witness mismatch")
    if abstract_partition_raw.get("task_instances_per_pair") != 20:
        raise HistoricalExposureError(
            "Abstract pair partition must bind 20 task instances per pair"
        )
    if abstract_partition_raw.get("task_counts") != {
        name: count * 20 for name, count in abstract_partition.counts.items()
    }:
        raise HistoricalExposureError("Abstract pair partition task counts mismatch")
    if set(abstract_partition.eligible) & set(pairs):
        raise HistoricalExposureError(
            "Abstract pair partition includes a historically exposed pair"
        )

    inventory = _require_object(body.get("generator_inventory"), label="generator_inventory")
    if inventory.get("basic_shape_count") != EXPECTED_SHAPE_COUNT:
        raise HistoricalExposureError("Basic generator inventory count mismatch")
    if inventory.get("abstract_attribute_count") != EXPECTED_ATTRIBUTE_COUNT:
        raise HistoricalExposureError("Abstract attribute inventory count mismatch")
    if inventory.get("admissible_abstract_pair_count") != len(admissible_pairs):
        raise HistoricalExposureError("admissible Abstract pair count mismatch")
    if len(admissible_pairs) != EXPECTED_ADMISSIBLE_PAIR_COUNT:
        raise HistoricalExposureError("admissible Abstract pair universe must contain 194 pairs")
    if inventory.get("admissible_abstract_pair_digest") != _line_digest(
        "\t".join(pair) for pair in admissible_pairs
    ):
        raise HistoricalExposureError("admissible Abstract pair digest mismatch")
    if not set(pairs) <= set(admissible_pairs):
        raise HistoricalExposureError("exposed Abstract pair is absent from admissible universe")
    expected_abstract_partition = _partition_abstract_pairs(admissible_pairs, pairs)
    if abstract_partition != expected_abstract_partition:
        raise HistoricalExposureError(
            "Abstract pair partition differs from deterministic hash ranking"
        )
    expected_unused_pairs = set(admissible_pairs) - set(pairs)
    if set(abstract_partition.eligible) != expected_unused_pairs:
        missing = sorted(expected_unused_pairs - set(abstract_partition.eligible))
        extra = sorted(set(abstract_partition.eligible) - expected_unused_pairs)
        raise HistoricalExposureError(
            "Abstract pair partition does not exactly exhaust the unused universe: "
            f"missing={missing}, extra={extra}"
        )

    freeform_status = freeform.get("status")
    if freeform_status != "indeterminate" or freeform.get("semantic_family_partition_available") is not False:
        raise HistoricalExposureError("Freeform exposure must remain indeterminate")
    freeform_tasks = _unique_strings(
        freeform.get("exact_task_ids"), label="freeform.exact_task_ids"
    )

    evidence_values = body.get("evidence_files")
    if not isinstance(evidence_values, list):
        raise HistoricalExposureError("evidence_files must be a list")
    evidence: list[tuple[str, str]] = []
    for record in evidence_values:
        item = _require_object(record, label="evidence file")
        path = item.get("path")
        sha256 = item.get("sha256")
        if not isinstance(path, str) or not path or not isinstance(sha256, str) or not sha256.startswith("sha256:"):
            raise HistoricalExposureError("evidence file lacks canonical path/sha256")
        evidence.append((path, sha256))
    if len(evidence) != len({path for path, _digest in evidence}):
        raise HistoricalExposureError("evidence_files contains duplicate paths")

    return HistoricalExposureSeed(
        exact_official_task_ids=tuple(exact_tasks),
        exact_official_panel_ids=panels,
        basic_shape_families=basic_shapes,
        visual_basic_shape_families=visual_basic,
        abstract_attributes=attributes,
        visual_abstract_attributes=visual_attributes,
        abstract_pairs=tuple(pairs),
        freeform_status=freeform_status,
        freeform_exact_task_ids=freeform_tasks,
        partition=partition,
        abstract_partition=abstract_partition,
        admissible_abstract_pairs=admissible_pairs,
        evidence_files=tuple(evidence),
        seed_digest=raw["seed_digest"],
        raw=body,
    )


def build_historical_exposure(
    repo_root: str | Path,
    dataset_root: str | Path | None = None,
    *,
    _evidence_reader: _RepositoryEvidenceReader | None = None,
) -> dict[str, Any]:
    """Reconstruct the seed from audited artifacts without opening any PNG.

    Ambiguous sources are rejected.  The function reads JSON, Markdown, Python
    source, and TSV metadata only; it never imports a sampler or renders/views
    an official corpus panel.
    """

    root = Path(repo_root).expanduser().resolve()
    dataset = (
        Path(dataset_root).expanduser().resolve()
        if dataset_root is not None
        else root / "downloads/Bongard-LOGO"
    )
    reader = _evidence_reader or _RepositoryEvidenceReader(root)
    body, _partition, _abstract_partition, _pairs = _build_seed_body(
        root, dataset, reader
    )
    return {"seed": body, "seed_digest": _address(body)}


def load_historical_exposure(
    path: str | Path = DEFAULT_SEED_PATH,
    *,
    repo_root: str | Path | None = None,
    dataset_root: str | Path | None = None,
    verify_evidence: bool = False,
    _evidence_reader: _RepositoryEvidenceReader | None = None,
) -> HistoricalExposureSeed:
    """Load and verify the self-contained persisted seed.

    Normal benchmark setup deliberately does not read the legacy ``crack_lab``
    tree.  Set ``verify_evidence=True`` (or call
    :func:`verify_historical_exposure`) only for an explicit historical audit.
    """

    seed_path = Path(path).expanduser().resolve()
    raw = _load_json_object(seed_path, label="historical exposure seed")
    _require_fields(raw, {"seed", "seed_digest"}, label="historical exposure envelope")
    serialized_body = _require_object(raw["seed"], label="historical exposure seed")
    if raw["seed_digest"] != _address(serialized_body):
        raise HistoricalExposureError("historical exposure seed digest mismatch")
    root = (
        Path(repo_root).expanduser().resolve()
        if repo_root is not None
        else Path(__file__).resolve().parent.parent
    )
    dataset = (
        Path(dataset_root).expanduser().resolve()
        if dataset_root is not None
        else root / "downloads/Bongard-LOGO"
    )

    if verify_evidence:
        evidence_values = serialized_body.get("evidence_files")
        if not isinstance(evidence_values, list):
            raise HistoricalExposureError("evidence_files must be a list")
        expected_digests: dict[str, str] = {}
        for record in evidence_values:
            item = _require_object(record, label="evidence file")
            relative_path = item.get("path")
            digest = item.get("sha256")
            if (
                not isinstance(relative_path, str)
                or not relative_path
                or not isinstance(digest, str)
                or not digest.startswith("sha256:")
            ):
                raise HistoricalExposureError("evidence file lacks canonical path/sha256")
            if relative_path in expected_digests:
                raise HistoricalExposureError("evidence_files contains duplicate paths")
            expected_digests[relative_path] = digest
        specified_paths = {path for path, _purpose, _commit in _EVIDENCE_SPECS}
        if set(expected_digests) != specified_paths:
            raise HistoricalExposureError("persisted evidence paths differ from the audit specification")

        reader = _evidence_reader or _RepositoryEvidenceReader(
            root, expected_digests=expected_digests
        )
        if reader.repo_root.resolve() != root:
            raise HistoricalExposureError("injected evidence reader belongs to a different repository")
        if reader.expected_digests != expected_digests:
            raise HistoricalExposureError("injected evidence reader lacks the exact seed hash map")
        rebuilt = build_historical_exposure(
            root,
            dataset,
            _evidence_reader=reader,
        )
        if raw != rebuilt:
            raise HistoricalExposureError(
                "persisted historical seed differs from the audited repository evidence"
            )
        body, partition, abstract_partition, admissible_pairs = _build_seed_body(
            root, dataset, reader
        )
        if body != rebuilt["seed"]:
            raise HistoricalExposureError("non-deterministic historical exposure reconstruction")
    else:
        body = serialized_body
        partition_raw = _require_object(body.get("basic_partition"), label="basic_partition")
        members = _require_object(partition_raw.get("members"), label="basic_partition.members")
        drill = _unique_strings(members.get("drill"), label="basic_partition.members.drill")
        dev = _unique_strings(members.get("dev"), label="basic_partition.members.dev")
        sealed = _unique_strings(members.get("sealed"), label="basic_partition.members.sealed")
        if set(drill) & set(dev) or set(drill) & set(sealed) or set(dev) & set(sealed):
            raise HistoricalExposureError("self-contained Basic partition members overlap")
        partition = BasicFamilyPartition(
            namespace=partition_raw.get("namespace"),
            drill=drill,
            dev=dev,
            sealed=sealed,
            drill_digest=_line_digest(drill),
            dev_digest=_line_digest(dev),
            sealed_digest=_line_digest(sealed),
        )
        inventory = _require_object(body.get("generator_inventory"), label="generator_inventory")
        pair_values = inventory.get("admissible_abstract_pairs")
        if not isinstance(pair_values, list):
            raise HistoricalExposureError("admissible_abstract_pairs must be a list")
        parsed_pairs: list[tuple[str, str]] = []
        for value in pair_values:
            if (
                not isinstance(value, list)
                or len(value) != 2
                or not all(isinstance(item, str) and item for item in value)
                or value[0] == value[1]
            ):
                raise HistoricalExposureError(f"invalid admissible Abstract pair: {value!r}")
            parsed_pairs.append((value[0], value[1]))
        admissible_pairs = tuple(parsed_pairs)
        if len(admissible_pairs) != len(set(admissible_pairs)):
            raise HistoricalExposureError("admissible_abstract_pairs contains duplicates")

        abstract_partition_raw = _require_object(
            body.get("abstract_pair_partition"), label="abstract_pair_partition"
        )
        abstract_members = _require_object(
            abstract_partition_raw.get("members"),
            label="abstract_pair_partition.members",
        )
        abstract_drill = _unique_pairs(
            abstract_members.get("drill"),
            label="abstract_pair_partition.members.drill",
        )
        abstract_dev = _unique_pairs(
            abstract_members.get("dev"),
            label="abstract_pair_partition.members.dev",
        )
        abstract_sealed = _unique_pairs(
            abstract_members.get("sealed"),
            label="abstract_pair_partition.members.sealed",
        )
        abstract_partition = AbstractPairPartition(
            namespace=abstract_partition_raw.get("namespace"),
            drill=abstract_drill,
            dev=abstract_dev,
            sealed=abstract_sealed,
            drill_digest=_line_digest(
                _pair_identifier(pair) for pair in abstract_drill
            ),
            dev_digest=_line_digest(_pair_identifier(pair) for pair in abstract_dev),
            sealed_digest=_line_digest(
                _pair_identifier(pair) for pair in abstract_sealed
            ),
        )

    return _seed_from_raw(
        raw,
        partition=partition,
        abstract_partition=abstract_partition,
        admissible_pairs=admissible_pairs,
    )


def verify_historical_exposure(
    path: str | Path = DEFAULT_SEED_PATH,
    *,
    repo_root: str | Path | None = None,
    dataset_root: str | Path | None = None,
    _evidence_reader: _RepositoryEvidenceReader | None = None,
) -> HistoricalExposureSeed:
    """Explicit verification alias used by benchmark setup code."""

    return load_historical_exposure(
        path,
        repo_root=repo_root,
        dataset_root=dataset_root,
        verify_evidence=True,
        _evidence_reader=_evidence_reader,
    )


__all__ = [
    "ABSTRACT_PAIR_PARTITION_NAMESPACE",
    "AbstractPairPartition",
    "BASIC_PARTITION_NAMESPACE",
    "BasicFamilyPartition",
    "DEFAULT_SEED_PATH",
    "HistoricalExposureError",
    "HistoricalExposureSeed",
    "build_historical_exposure",
    "load_historical_exposure",
    "verify_historical_exposure",
]

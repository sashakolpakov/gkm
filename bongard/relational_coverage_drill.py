"""Offline, candidate-independent coverage drilling for relational witnesses.

The command in this module has one deliberately narrow job: choose a bounded,
deterministic pilot from exact-unused ShapeBongard V2 train/validation tasks and
measure how often the Python loop-scene extractor can produce its typed
geometry and contact observations.  It never reads action-program JSON, never
calls a proposer or another model, and never authorizes official test pixels.

Selection is metadata-only.  A successor exposure ledger is written, fsynced,
and reloaded before the first selected PNG path is resolved, opened, hashed, or
decoded.  Only selected tasks receive PNG manifests.  The ``generator`` key is
an engineering stratum derived solely from the task ID; it is not a claim of
semantic independence.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Any, Callable, Iterable, Mapping, Sequence

from bongard.artifacts import canonical_json
from bongard.cohorts import classify_task, parse_official_task_id
from bongard.corpus import FAMILIES, PNG_SIGNATURE, SplitIndex
from bongard.evidence import Disposition
from bongard.exposure import (
    ExposureLedger,
    ExposureViolation,
    SemanticDisclosureKey,
    basic_morphology_cluster_id,
    semantic_resolver_policy_digest,
)
from bongard.historical_exposure import load_historical_exposure
from bongard.loop_geometry import (
    LOOP_GEOMETRY_ALGORITHM_ID,
    loop_geometry_algorithm_digest,
    loop_geometry_source_digest,
)
from bongard.loop_scene_witnesses import (
    LOOP_SCENE_ALGORITHM_ID,
    LoopScenePacket,
    extract_loop_scene_witnesses,
    loop_scene_catalog_digest,
    loop_scene_extractor_digest,
)
from bongard.point_contact import (
    POINT_CONTACT_ALGORITHM_ID,
    point_contact_algorithm_digest,
    point_contact_source_digest,
)
from bongard.visual_witness_bundle import (
    VISUAL_WITNESS_BUNDLE_ALGORITHM_ID,
    visual_witness_bundle_catalog_digest,
    visual_witness_bundle_extractor_digest,
)


SCHEMA = "gkm.bongard-relational-coverage-drill.v2"
SELECTION_SCHEMA = "gkm.bongard-relational-coverage-selection.v2"
SELECTED_MANIFEST_SCHEMA = "gkm.bongard-selected-png-manifest.v2"
ALGORITHM_ID = "bongard.relational-coverage-drill/hash-stratified-v2"
DEFAULT_NAMESPACE = "bongard-relational-coverage-pilot-v2"
STRICT_DEV_CAPACITY_REFERENCE = 16

_ADDRESS = re.compile(r"sha256:[0-9a-f]{64}\Z")
_TASK_ID = re.compile(r"^(ff|bd|hd)_(.+)_([0-9]{4})\Z")


class CoverageDrillError(RuntimeError):
    """A coverage run violated its selection, exposure, or input boundary."""


def _address(value: object) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(value)).hexdigest()


def _bytes_address(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _require_address(value: str, label: str) -> str:
    if not isinstance(value, str) or _ADDRESS.fullmatch(value) is None:
        raise CoverageDrillError(f"{label} must be a prefixed lowercase SHA-256")
    return value


def _positive_int(value: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CoverageDrillError(f"{label} must be a positive integer")
    return value


def _nonnegative_int(value: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CoverageDrillError(f"{label} must be a non-negative integer")
    return value


def _rank(namespace: str, stage: str, *parts: str) -> str:
    payload = "\0".join((namespace, stage, *parts)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def task_generator_key(task_id: str) -> tuple[str, str]:
    """Return ``(family, metadata-only generator stratum)`` from one task ID.

    FF strata are the repeated ``nact`` program families.  BD and HD strata
    are their concept-expression prefixes.  Some BD strata are singletons;
    the second-stage family cap is what keeps that fact from expanding a pilot
    into the entire BD split.
    """

    if not isinstance(task_id, str):
        raise CoverageDrillError("task ID must be text")
    match = _TASK_ID.fullmatch(task_id)
    if match is None:
        raise CoverageDrillError(f"malformed ShapeBongard task ID: {task_id!r}")
    family, expression, _ = match.groups()
    if not expression or "\0" in expression:
        raise CoverageDrillError("task generator expression is malformed")
    return family, expression


@dataclass(frozen=True, slots=True)
class SelectedTask:
    task_id: str
    family: str
    split: str
    generator: str
    generator_rank: str
    family_rank: str

    def to_data(self) -> dict[str, str]:
        return {
            "task_id": self.task_id,
            "family": self.family,
            "split": self.split,
            "generator": self.generator,
            "generator_rank": self.generator_rank,
            "family_rank": self.family_rank,
        }


@dataclass(frozen=True, slots=True)
class CoverageSelection:
    namespace: str
    per_generator: int
    per_split_family: int
    source_corpus_manifest_digest: str
    split_source_digest: str
    exposure_predecessor_digest: str
    exact_unused_count: int
    protected_strict_dev_task_ids: tuple[str, ...]
    protected_strict_dev_semantic_keys: tuple[
        tuple[str, tuple[str, ...]], ...
    ]
    protected_strict_dev_disclosure_tokens: tuple[str, ...]
    protected_strict_dev_closure_task_ids: tuple[str, ...]
    historical_seed_digest: str
    semantic_resolver_policy_digest: str
    minimum_strict_dev_reserve: int
    postselection_strict_dev_task_count: int
    generator_shortlist_count: int
    selected: tuple[SelectedTask, ...]
    digest: str

    def content_data(self) -> dict[str, object]:
        return {
            "schema": SELECTION_SCHEMA,
            "algorithm_id": ALGORITHM_ID,
            "namespace": self.namespace,
            "allowed_splits": ["train", "val"],
            "per_generator": self.per_generator,
            "per_split_family": self.per_split_family,
            "source_corpus_manifest_digest": self.source_corpus_manifest_digest,
            "split_source_digest": self.split_source_digest,
            "exposure_predecessor_digest": self.exposure_predecessor_digest,
            "exact_unused_count": self.exact_unused_count,
            "strict_dev_protection": {
                "policy": (
                    "exclude the complete metadata-only disclosure-token closure "
                    "of every individually viable frozen DEV task; Basic tokens "
                    "include exact family and morphology cluster, and Abstract "
                    "tokens include exact pair and each constituent attribute"
                ),
                "reference_capacity_after_a3": STRICT_DEV_CAPACITY_REFERENCE,
                "minimum_reserved": self.minimum_strict_dev_reserve,
                "historical_seed_digest": self.historical_seed_digest,
                "semantic_resolver_policy_digest": (
                    self.semantic_resolver_policy_digest
                ),
                "protected_task_count": len(self.protected_strict_dev_task_ids),
                "protected_task_ids_digest": _address(
                    list(self.protected_strict_dev_task_ids)
                ),
                "protected_semantic_keys": [
                    {"kind": kind, "concepts": list(concepts)}
                    for kind, concepts in self.protected_strict_dev_semantic_keys
                ],
                "protected_semantic_keys_digest": _address(
                    [
                        {"kind": kind, "concepts": list(concepts)}
                        for kind, concepts in self.protected_strict_dev_semantic_keys
                    ]
                ),
                "protected_disclosure_tokens": list(
                    self.protected_strict_dev_disclosure_tokens
                ),
                "protected_disclosure_tokens_digest": _address(
                    list(self.protected_strict_dev_disclosure_tokens)
                ),
                "semantic_closure_task_count": len(
                    self.protected_strict_dev_closure_task_ids
                ),
                "semantic_closure_task_ids_digest": _address(
                    list(self.protected_strict_dev_closure_task_ids)
                ),
                "capacity_preservation": {
                    "baseline_individually_viable_task_count": len(
                        self.protected_strict_dev_task_ids
                    ),
                    "postselection_individually_viable_task_count": (
                        self.postselection_strict_dev_task_count
                    ),
                    "all_baseline_tasks_remain_viable": (
                        self.postselection_strict_dev_task_count
                        == len(self.protected_strict_dev_task_ids)
                    ),
                    "selected_tokens_disjoint_from_protected_closure": True,
                },
            },
            "generator_stratification_qualification": (
                "task-ID-derived engineering coverage strata; not evidence of "
                "semantic independence"
            ),
            "generator_shortlist_count": self.generator_shortlist_count,
            "selected": [item.to_data() for item in self.selected],
        }

    def to_data(self) -> dict[str, object]:
        result = self.content_data()
        result["digest"] = self.digest
        return result


@dataclass(frozen=True, slots=True)
class _StrictDevProtection:
    task_ids: tuple[str, ...]
    semantic_keys: tuple[SemanticDisclosureKey, ...]
    disclosure_tokens: tuple[str, ...]
    closure_task_ids: tuple[str, ...]
    historical_seed_digest: str
    resolver_policy_digest: str


def _task_semantic_keys(
    task_id: str, historical: Any
) -> tuple[SemanticDisclosureKey, ...]:
    """Derive public resolver keys even for a task blocked on another key.

    ``assert_semantically_unseen`` intentionally fails without returning a
    resolution on any collision.  Closure construction must still see every
    key of a mixed task: the completed v1 pilot selected a Basic pair already
    blocked on one concept which nevertheless disclosed a reserved DEV concept.
    """

    parsed = parse_official_task_id(task_id, historical)
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
    return (SemanticDisclosureKey("freeform_family", parsed.concepts),)


def _task_disclosure_tokens(task_id: str, historical: Any) -> tuple[str, ...]:
    """Return the conservative strict-DEV collision tokens for one task."""

    parsed = parse_official_task_id(task_id, historical)
    if parsed.family == "bd":
        return tuple(
            sorted(
                {
                    token
                    for concept in parsed.concepts
                    for token in (
                        "basic_family:" + concept,
                        "basic_morphology:" + basic_morphology_cluster_id(concept),
                    )
                }
            )
        )
    if parsed.family == "hd":
        return tuple(
            sorted(
                {"abstract_pair:" + "\0".join(parsed.concepts)}
                | {
                    "abstract_attribute:" + concept
                    for concept in parsed.concepts
                }
            )
        )
    return ("freeform_family:" + "\0".join(parsed.concepts),)


def _ledger_exposed_hd_attributes(
    predecessor: ExposureLedger, historical: Any
) -> frozenset[str]:
    attributes: set[str] = set()
    for task_id in predecessor.exposed_task_ids:
        parsed = parse_official_task_id(task_id, historical)
        if parsed.family == "hd":
            attributes.update(parsed.concepts)
    return frozenset(attributes)


def _strict_dev_protection(
    task_ids: Iterable[str], predecessor: ExposureLedger
) -> _StrictDevProtection:
    """Protect the full disclosure closure of every viable strict DEV task."""

    historical = load_historical_exposure()
    resolver = semantic_resolver_policy_digest(historical)
    exposed_hd_attributes = _ledger_exposed_hd_attributes(predecessor, historical)
    protected: list[str] = []
    semantic_keys: set[SemanticDisclosureKey] = set()
    tokens: set[str] = set()
    ordered_task_ids = tuple(sorted(task_ids))
    for task_id in ordered_task_ids:
        record = classify_task(task_id, historical)
        if not (
            record.historically_clean and record.semantic_cohort == "dev"
        ):
            continue
        if record.family == "hd" and (
            set(record.parsed.concepts) & exposed_hd_attributes
        ):
            continue
        try:
            resolution = predecessor.assert_semantically_unseen(
                task_ids=(task_id,),
                historical_seed=historical,
                expected_historical_seed_digest=historical.seed_digest,
                expected_resolver_policy_digest=resolver,
            )
        except ExposureViolation:
            continue
        expected_keys = _task_semantic_keys(task_id, historical)
        if resolution.semantic_keys != expected_keys:
            raise CoverageDrillError(
                "coverage disclosure-key derivation differs from the bound "
                f"semantic resolver for {task_id}"
            )
        protected.append(task_id)
        semantic_keys.update(resolution.semantic_keys)
        tokens.update(_task_disclosure_tokens(task_id, historical))
    closure = tuple(
        task_id
        for task_id in ordered_task_ids
        if set(_task_disclosure_tokens(task_id, historical)) & tokens
    )
    return _StrictDevProtection(
        task_ids=tuple(protected),
        semantic_keys=tuple(sorted(semantic_keys)),
        disclosure_tokens=tuple(sorted(tokens)),
        closure_task_ids=closure,
        historical_seed_digest=historical.seed_digest,
        resolver_policy_digest=resolver,
    )


def _empty_strict_dev_protection() -> _StrictDevProtection:
    historical = load_historical_exposure()
    return _StrictDevProtection(
        task_ids=(),
        semantic_keys=(),
        disclosure_tokens=(),
        closure_task_ids=(),
        historical_seed_digest=historical.seed_digest,
        resolver_policy_digest=semantic_resolver_policy_digest(historical),
    )


def _verify_strict_dev_protection_after_selection(
    *,
    predecessor: ExposureLedger,
    protection: _StrictDevProtection,
    selected_task_ids: Sequence[str],
) -> int:
    """Prove metadata-only that the pilot cannot consume baseline DEV slots."""

    if not protection.task_ids:
        return 0
    historical = load_historical_exposure()
    if historical.seed_digest != protection.historical_seed_digest:
        raise CoverageDrillError("historical semantic seed changed during selection")
    if (
        semantic_resolver_policy_digest(historical)
        != protection.resolver_policy_digest
    ):
        raise CoverageDrillError("semantic resolver policy changed during selection")
    selected_tokens = {
        token
        for task_id in selected_task_ids
        for token in _task_disclosure_tokens(task_id, historical)
    }
    collisions = selected_tokens & set(protection.disclosure_tokens)
    if collisions:
        raise CoverageDrillError(
            "selected tasks intersect the protected strict DEV disclosure closure: "
            f"{sorted(collisions)}"
        )

    simulated = predecessor.record(
        phase="relational-coverage-selection-simulation-v2",
        actor="bongard.relational_coverage_drill",
        purpose="metadata-only strict DEV capacity preservation proof",
        task_ids=selected_task_ids,
        source=ALGORITHM_ID,
        observed_at="1970-01-01T00:00:00Z",
        require_unseen=True,
    )
    exposed_hd_attributes = _ledger_exposed_hd_attributes(simulated, historical)
    postselection_viable = 0
    failures: list[str] = []
    for task_id in protection.task_ids:
        parsed = parse_official_task_id(task_id, historical)
        if parsed.family == "hd" and (
            set(parsed.concepts) & exposed_hd_attributes
        ):
            failures.append(task_id)
            continue
        try:
            simulated.assert_semantically_unseen(
                task_ids=(task_id,),
                historical_seed=historical,
                expected_historical_seed_digest=(
                    protection.historical_seed_digest
                ),
                expected_resolver_policy_digest=(
                    protection.resolver_policy_digest
                ),
            )
        except ExposureViolation:
            failures.append(task_id)
            continue
        postselection_viable += 1
    if failures or postselection_viable != len(protection.task_ids):
        raise CoverageDrillError(
            "selected task IDs reduce baseline strict DEV capacity; "
            f"invalidated={failures}"
        )
    return postselection_viable


def select_exact_unused_pilot(
    split_index: SplitIndex,
    predecessor: ExposureLedger,
    *,
    source_corpus_manifest_digest: str,
    namespace: str = DEFAULT_NAMESPACE,
    per_generator: int = 1,
    per_split_family: int = 12,
    minimum_strict_dev_reserve: int = STRICT_DEV_CAPACITY_REFERENCE,
    require_official_split: bool = True,
) -> CoverageSelection:
    """Select a bounded hash-ranked pilot without touching any panel path."""

    if not isinstance(split_index, SplitIndex) or not split_index.groups:
        raise CoverageDrillError("coverage selection requires a non-empty SplitIndex")
    if split_index.source_digest is None:
        raise CoverageDrillError("coverage selection requires an authenticated split")
    _require_address(split_index.source_digest, "split source digest")
    source = _require_address(
        source_corpus_manifest_digest, "source corpus manifest digest"
    )
    if not isinstance(predecessor, ExposureLedger):
        raise TypeError("predecessor must be an ExposureLedger")
    predecessor.assert_corpus(source)
    if not isinstance(namespace, str) or not namespace.strip() or "\0" in namespace:
        raise CoverageDrillError("selection namespace must be non-empty text")
    per_generator = _positive_int(per_generator, "per_generator")
    per_split_family = _positive_int(
        per_split_family, "per_split_family"
    )
    minimum_strict_dev_reserve = _nonnegative_int(
        minimum_strict_dev_reserve, "minimum_strict_dev_reserve"
    )

    groups = split_index.canonical_groups
    referenced = set(groups["train"]) | set(groups["val"]) | set(groups["test"])
    split_index.validate(referenced, official_counts=require_official_split)
    allowed = set(groups["train"]) | set(groups["val"])
    if allowed & set(groups["test"]):
        raise CoverageDrillError("official test IDs overlap train/validation")
    exact_unused = predecessor.unseen_task_ids(allowed)
    protection = (
        _strict_dev_protection(exact_unused, predecessor)
        if minimum_strict_dev_reserve
        else _empty_strict_dev_protection()
    )
    if len(protection.task_ids) < minimum_strict_dev_reserve:
        raise CoverageDrillError(
            "strict DEV reserve has fallen below the requested minimum: "
            f"{len(protection.task_ids)} < {minimum_strict_dev_reserve}"
        )
    eligible = tuple(
        sorted(set(exact_unused) - set(protection.closure_task_ids))
    )

    by_generator: dict[tuple[str, str, str], list[SelectedTask]] = defaultdict(list)
    for task_id in eligible:
        assignment = split_index.assignment(task_id)
        if assignment.split not in {"train", "val"} or assignment.regime is not None:
            raise CoverageDrillError(
                f"eligible task is not exclusively train/validation: {task_id}"
            )
        family, generator = task_generator_key(task_id)
        generator_rank = _rank(
            namespace,
            "within-generator",
            assignment.split,
            family,
            generator,
            task_id,
        )
        family_rank = _rank(
            namespace, "within-split-family", assignment.split, family, task_id
        )
        by_generator[(assignment.split, family, generator)].append(
            SelectedTask(
                task_id=task_id,
                family=family,
                split=assignment.split,
                generator=generator,
                generator_rank=generator_rank,
                family_rank=family_rank,
            )
        )

    shortlisted: list[SelectedTask] = []
    for key in sorted(by_generator):
        ranked = sorted(
            by_generator[key], key=lambda item: (item.generator_rank, item.task_id)
        )
        shortlisted.extend(ranked[:per_generator])

    by_family: dict[tuple[str, str], list[SelectedTask]] = defaultdict(list)
    for item in shortlisted:
        by_family[(item.split, item.family)].append(item)
    selected: list[SelectedTask] = []
    for key in sorted(by_family):
        ranked = sorted(
            by_family[key], key=lambda item: (item.family_rank, item.task_id)
        )
        selected.extend(ranked[:per_split_family])
    selected_tuple = tuple(
        sorted(
            selected,
            key=lambda item: (item.split, item.family, item.generator, item.task_id),
        )
    )
    if not selected_tuple:
        raise CoverageDrillError("no exact-unused train/validation tasks are selectable")

    postselection_strict_dev_task_count = (
        _verify_strict_dev_protection_after_selection(
            predecessor=predecessor,
            protection=protection,
            selected_task_ids=tuple(item.task_id for item in selected_tuple),
        )
    )

    provisional = CoverageSelection(
        namespace=namespace,
        per_generator=per_generator,
        per_split_family=per_split_family,
        source_corpus_manifest_digest=source,
        split_source_digest=split_index.source_digest,
        exposure_predecessor_digest=predecessor.digest,
        exact_unused_count=len(exact_unused),
        protected_strict_dev_task_ids=protection.task_ids,
        protected_strict_dev_semantic_keys=tuple(
            (key.kind, key.concepts) for key in protection.semantic_keys
        ),
        protected_strict_dev_disclosure_tokens=protection.disclosure_tokens,
        protected_strict_dev_closure_task_ids=protection.closure_task_ids,
        historical_seed_digest=protection.historical_seed_digest,
        semantic_resolver_policy_digest=protection.resolver_policy_digest,
        minimum_strict_dev_reserve=minimum_strict_dev_reserve,
        postselection_strict_dev_task_count=(
            postselection_strict_dev_task_count
        ),
        generator_shortlist_count=len(shortlisted),
        selected=selected_tuple,
        digest="",
    )
    return replace(provisional, digest=_address(provisional.content_data()))


def _write_once_durable(path: Path, payload: bytes) -> Path:
    """Write immutable bytes, fsync the file and directory, then verify."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(destination, flags, 0o600)
    except FileExistsError:
        if destination.read_bytes() != payload:
            raise CoverageDrillError(
                f"refusing to overwrite different artifact at {destination}"
            )
    else:
        try:
            view = memoryview(payload)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise CoverageDrillError(f"short write to {destination}")
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    directory = os.open(destination.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    if destination.read_bytes() != payload:
        raise CoverageDrillError(f"durable artifact verification failed: {destination}")
    return destination


def _persist_successor(successor: ExposureLedger, directory: Path) -> Path:
    filename = successor.digest.removeprefix("sha256:") + ".exposure.json"
    path = _write_once_durable(
        Path(directory) / filename, successor.to_json().encode("utf-8")
    )
    if ExposureLedger.load(path) != successor:
        raise CoverageDrillError("durable exposure successor failed cold reload")
    return path


def _read_png_no_follow(path: Path) -> bytes:
    """Read one regular PNG from a stable no-follow file descriptor."""

    path = Path(path)
    try:
        before = os.lstat(path)
    except OSError as exc:
        raise CoverageDrillError(f"cannot inspect selected panel: {path}") from exc
    if not stat.S_ISREG(before.st_mode):
        raise CoverageDrillError(f"selected panel is not a regular file: {path}")
    identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise CoverageDrillError(f"cannot open selected panel safely: {path}") from exc
    chunks: list[bytes] = []
    try:
        opened = os.fstat(descriptor)
        opened_identity = (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
            opened.st_mtime_ns,
            opened.st_ctime_ns,
        )
        if not stat.S_ISREG(opened.st_mode) or opened_identity != identity:
            raise CoverageDrillError(f"selected panel changed while opening: {path}")
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        after_identity = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if after_identity != identity:
            raise CoverageDrillError(f"selected panel changed while reading: {path}")
    finally:
        os.close(descriptor)
    payload = b"".join(chunks)
    try:
        final = os.lstat(path)
    except OSError as exc:
        raise CoverageDrillError(f"selected panel disappeared: {path}") from exc
    final_identity = (
        final.st_dev,
        final.st_ino,
        final.st_size,
        final.st_mtime_ns,
        final.st_ctime_ns,
    )
    if final_identity != identity or len(payload) != before.st_size:
        raise CoverageDrillError(f"selected panel changed after reading: {path}")
    if not payload.startswith(PNG_SIGNATURE):
        raise CoverageDrillError(f"selected file lacks the PNG signature: {path}")
    return payload


def _task_panel_paths(corpus_root: Path, selected: SelectedTask) -> tuple[tuple[str, int, Path], ...]:
    """Resolve only one already-precommitted task's fourteen panel paths."""

    family_root = Path(corpus_root) / selected.family
    candidates = tuple(
        family_root / component / selected.task_id for component in ("images", "png")
    )
    present = tuple(path for path in candidates if path.is_dir())
    if len(present) != 1:
        raise CoverageDrillError(
            f"selected task has {len(present)} usable image layouts: {selected.task_id}"
        )
    task_root = present[0]
    result: list[tuple[str, int, Path]] = []
    for polarity, label in (("positive", "1"), ("negative", "0")):
        for index in range(7):
            result.append((polarity, index, task_root / label / f"{index}.png"))
    return tuple(result)


def _bucket(value: int, boundaries: Sequence[int]) -> str:
    lower = 0
    for upper in boundaries:
        if value < upper:
            return f"{lower}-{upper - 1}"
        lower = upper
    return f"{lower}+"


class _Counters:
    def __init__(self) -> None:
        self.task_count = 0
        self.panel_count = 0
        self.panel_success_count = 0
        self.panel_error_count = 0
        self.panel_errors: Counter[str] = Counter()
        self.scenario_count = 0
        self.zero_loop_scenario_count = 0
        self.loop_count_histogram: Counter[str] = Counter()
        self.panel_scenario_loop_spread: Counter[str] = Counter()
        self.loop_count = 0
        self.loop_area_buckets: Counter[str] = Counter()
        self.boundary_cycle_count: Counter[str] = Counter()
        self.loop_owner: Counter[str] = Counter()
        self.substantiveness_dispositions: Counter[str] = Counter()
        self.substantiveness_reasons: Counter[str] = Counter()
        self.nuisance_or_unresolved_loop_count = 0
        self.polygon_dispositions: Counter[str] = Counter()
        self.polygon_reasons: Counter[str] = Counter()
        self.polygon_side_intervals: Counter[str] = Counter()
        self.polygon_variant_count: Counter[str] = Counter()
        self.obliqueness_dispositions: Counter[str] = Counter()
        self.obliqueness_reasons: Counter[str] = Counter()
        self.obliqueness_minimum_buckets: Counter[str] = Counter()
        self.contact_count = 0
        self.contact_dispositions: Counter[str] = Counter()
        self.contact_kinds: Counter[str] = Counter()
        self.contact_reasons: Counter[str] = Counter()
        self.contact_errors: Counter[str] = Counter()
        self.contact_gap_buckets: Counter[str] = Counter()
        self.contact_spread_buckets: Counter[str] = Counter()

    def record_task(self) -> None:
        self.task_count += 1

    def record_panel_error(self, exc: Exception) -> None:
        self.panel_count += 1
        self.panel_error_count += 1
        self.panel_errors[type(exc).__module__ + "." + type(exc).__qualname__] += 1

    def record_packet(self, packet: LoopScenePacket) -> None:
        self.panel_count += 1
        self.panel_success_count += 1
        loop_counts = [len(scenario.loops) for scenario in packet.scenarios]
        if loop_counts:
            self.panel_scenario_loop_spread[str(max(loop_counts) - min(loop_counts))] += 1
        for scenario in packet.scenarios:
            self.scenario_count += 1
            count = len(scenario.loops)
            self.loop_count_histogram[str(count)] += 1
            if count == 0:
                self.zero_loop_scenario_count += 1
            for loop in scenario.loops:
                self.loop_count += 1
                self.loop_area_buckets[_bucket(loop.area_pixels, (3, 9, 32, 128))] += 1
                self.boundary_cycle_count[str(loop.boundary_cycle_count)] += 1
                self.loop_owner["missing" if loop.owner_component_id is None else "present"] += 1
                substantive = loop.substantiveness
                self.substantiveness_dispositions[substantive.disposition.value] += 1
                self.substantiveness_reasons[substantive.reason_code] += 1
                if substantive.disposition is not Disposition.PRESENT:
                    self.nuisance_or_unresolved_loop_count += 1
                polygon = loop.polygon
                self.polygon_dispositions[polygon.disposition.value] += 1
                self.polygon_reasons[polygon.reason_code] += 1
                self.polygon_variant_count[str(len(polygon.variants))] += 1
                if polygon.side_count is not None:
                    key = f"{polygon.side_count.lower}:{polygon.side_count.upper}"
                    self.polygon_side_intervals[key] += 1
                oblique = loop.edge_obliqueness
                self.obliqueness_dispositions[oblique.disposition.value] += 1
                self.obliqueness_reasons[oblique.reason_code] += 1
                if oblique.minimum_millidegrees is not None:
                    value = oblique.minimum_millidegrees.lower
                    self.obliqueness_minimum_buckets[
                        _bucket(value, (5_000, 10_000, 15_000, 20_000, 30_000, 45_001))
                    ] += 1
            for contact in scenario.contacts:
                self.contact_count += 1
                self.contact_dispositions[contact.disposition.value] += 1
                self.contact_kinds[contact.contact_kind.value] += 1
                self.contact_reasons[contact.reason_code] += 1
                if contact.error_type is not None:
                    self.contact_errors[contact.error_type] += 1
                if contact.normalized_gap_ppm_upper is not None:
                    self.contact_gap_buckets[
                        _bucket(contact.normalized_gap_ppm_upper, (50_000, 100_000, 200_000, 300_001))
                    ] += 1
                if contact.interface_spread_ppm_upper is not None:
                    self.contact_spread_buckets[
                        _bucket(contact.interface_spread_ppm_upper, (50_000, 100_000, 200_000, 300_001))
                    ] += 1

    @staticmethod
    def _counter(value: Counter[str]) -> dict[str, int]:
        return {key: value[key] for key in sorted(value)}

    def to_data(self) -> dict[str, object]:
        return {
            "tasks": self.task_count,
            "extractor": {
                "panels_attempted": self.panel_count,
                "panels_succeeded": self.panel_success_count,
                "panels_errored": self.panel_error_count,
                "error_types": self._counter(self.panel_errors),
            },
            "scenarios": {
                "observed": self.scenario_count,
                "zero_loop": self.zero_loop_scenario_count,
                "loop_count_histogram": self._counter(self.loop_count_histogram),
                "per_panel_loop_count_spread_histogram": self._counter(
                    self.panel_scenario_loop_spread
                ),
            },
            "loops": {
                "observed": self.loop_count,
                "area_pixel_buckets": self._counter(self.loop_area_buckets),
                "boundary_cycle_count": self._counter(self.boundary_cycle_count),
                "owner_component": self._counter(self.loop_owner),
                "substantiveness_dispositions": self._counter(
                    self.substantiveness_dispositions
                ),
                "substantiveness_reasons": self._counter(
                    self.substantiveness_reasons
                ),
                "nuisance_or_unresolved": self.nuisance_or_unresolved_loop_count,
            },
            "polygon": {
                "dispositions": self._counter(self.polygon_dispositions),
                "reasons": self._counter(self.polygon_reasons),
                "side_count_intervals": self._counter(self.polygon_side_intervals),
                "variant_count": self._counter(self.polygon_variant_count),
            },
            "obliqueness": {
                "dispositions": self._counter(self.obliqueness_dispositions),
                "reasons": self._counter(self.obliqueness_reasons),
                "minimum_millidegree_lower_bound_buckets": self._counter(
                    self.obliqueness_minimum_buckets
                ),
            },
            "contact": {
                "observed_pairs": self.contact_count,
                "dispositions": self._counter(self.contact_dispositions),
                "kinds": self._counter(self.contact_kinds),
                "reasons": self._counter(self.contact_reasons),
                "error_types": self._counter(self.contact_errors),
                "normalized_gap_ppm_upper_buckets": self._counter(
                    self.contact_gap_buckets
                ),
                "interface_spread_ppm_upper_buckets": self._counter(
                    self.contact_spread_buckets
                ),
            },
        }


class _AggregateBook:
    def __init__(self) -> None:
        self.global_counter = _Counters()
        self.by_split: dict[str, _Counters] = defaultdict(_Counters)
        self.by_family: dict[str, _Counters] = defaultdict(_Counters)
        self.by_generator: dict[str, _Counters] = defaultdict(_Counters)
        self.by_split_family: dict[str, _Counters] = defaultdict(_Counters)
        self.by_polarity: dict[str, _Counters] = defaultdict(_Counters)

    def _for_task(self, item: SelectedTask) -> tuple[_Counters, ...]:
        return (
            self.global_counter,
            self.by_split[item.split],
            self.by_family[item.family],
            self.by_generator[f"{item.family}/{item.generator}"],
            self.by_split_family[f"{item.split}/{item.family}"],
        )

    def record_task(self, item: SelectedTask) -> None:
        for counter in self._for_task(item):
            counter.record_task()
        # Every Bongard task contributes seven panels of each polarity.
        self.by_polarity["positive"].record_task()
        self.by_polarity["negative"].record_task()

    def _for_panel(
        self, item: SelectedTask, polarity: str
    ) -> tuple[_Counters, ...]:
        if polarity not in {"positive", "negative"}:
            raise CoverageDrillError("panel polarity is not canonical")
        return (*self._for_task(item), self.by_polarity[polarity])

    def record_packet(
        self, item: SelectedTask, polarity: str, packet: LoopScenePacket
    ) -> None:
        for counter in self._for_panel(item, polarity):
            counter.record_packet(packet)

    def record_error(
        self, item: SelectedTask, polarity: str, exc: Exception
    ) -> None:
        for counter in self._for_panel(item, polarity):
            counter.record_panel_error(exc)

    @staticmethod
    def _map(value: Mapping[str, _Counters]) -> dict[str, object]:
        return {key: value[key].to_data() for key in sorted(value)}

    def to_data(self) -> dict[str, object]:
        return {
            "global": self.global_counter.to_data(),
            "by_split": self._map(self.by_split),
            "by_family": self._map(self.by_family),
            "by_generator": self._map(self.by_generator),
            "by_split_family": self._map(self.by_split_family),
            "by_polarity": self._map(self.by_polarity),
        }


def _algorithm_identities() -> dict[str, str]:
    source_digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return {
        "coverage_algorithm_id": ALGORITHM_ID,
        "coverage_python_source_digest": source_digest,
        "loop_scene_algorithm_id": LOOP_SCENE_ALGORITHM_ID,
        "loop_scene_catalog_digest": loop_scene_catalog_digest(),
        "loop_scene_extractor_digest": loop_scene_extractor_digest(),
        "loop_geometry_algorithm_id": LOOP_GEOMETRY_ALGORITHM_ID,
        "loop_geometry_algorithm_digest": loop_geometry_algorithm_digest(),
        "loop_geometry_python_source_digest": loop_geometry_source_digest(),
        "point_contact_algorithm_id": POINT_CONTACT_ALGORITHM_ID,
        "point_contact_algorithm_digest": point_contact_algorithm_digest(),
        "point_contact_python_source_digest": point_contact_source_digest(),
        "visual_witness_bundle_algorithm_id": VISUAL_WITNESS_BUNDLE_ALGORITHM_ID,
        "visual_witness_bundle_catalog_digest": visual_witness_bundle_catalog_digest(),
        "visual_witness_bundle_extractor_digest": visual_witness_bundle_extractor_digest(),
        "reference_execution": "python-canonical/v1",
    }


def _assert_packet_identity(
    packet: LoopScenePacket, algorithms: Mapping[str, str]
) -> None:
    expected = {
        "extractor_artifact_digest": algorithms["loop_scene_extractor_digest"],
        "loop_geometry_algorithm_digest": algorithms[
            "loop_geometry_algorithm_digest"
        ],
        "point_contact_algorithm_digest": algorithms[
            "point_contact_algorithm_digest"
        ],
        "parent_bundle_extractor_digest": algorithms[
            "visual_witness_bundle_extractor_digest"
        ],
    }
    changed = {
        field: (expected_value, getattr(packet, field))
        for field, expected_value in expected.items()
        if getattr(packet, field) != expected_value
    }
    if changed:
        raise CoverageDrillError(
            "Python extractor identity changed after the input commitment: "
            + ", ".join(sorted(changed))
        )


@dataclass(frozen=True, slots=True)
class CoverageDrillResult:
    report: Mapping[str, object]
    report_path: Path
    exposure_successor: ExposureLedger
    exposure_successor_path: Path


def run_coverage_drill(
    *,
    corpus_root: str | Path,
    split_index: SplitIndex,
    source_corpus_manifest_digest: str,
    predecessor: ExposureLedger,
    exposure_store: str | Path,
    output_store: str | Path,
    namespace: str = DEFAULT_NAMESPACE,
    per_generator: int = 1,
    per_split_family: int = 12,
    minimum_strict_dev_reserve: int = STRICT_DEV_CAPACITY_REFERENCE,
    require_official_split: bool = True,
    actor: str = "offline-relational-coverage-drill",
    observed_at: str | None = None,
    png_reader: Callable[[Path], bytes] = _read_png_no_follow,
    extractor: Callable[[bytes], LoopScenePacket] = extract_loop_scene_witnesses,
) -> CoverageDrillResult:
    """Persist exposure first, then extract all selected task panels."""

    root = Path(corpus_root).expanduser().resolve()
    if not root.is_dir():
        raise CoverageDrillError(f"corpus root is not a directory: {root}")
    selection = select_exact_unused_pilot(
        split_index,
        predecessor,
        source_corpus_manifest_digest=source_corpus_manifest_digest,
        namespace=namespace,
        per_generator=per_generator,
        per_split_family=per_split_family,
        minimum_strict_dev_reserve=minimum_strict_dev_reserve,
        require_official_split=require_official_split,
    )
    algorithms = _algorithm_identities()
    input_commitment = {
        "schema": "gkm.bongard-relational-coverage-input.v2",
        "source_corpus_manifest_digest": source_corpus_manifest_digest,
        "split_source_digest": split_index.source_digest,
        "exposure_predecessor_digest": predecessor.digest,
        "selection_digest": selection.digest,
        "algorithm_identities": algorithms,
        "restrictions": {
            "allowed_splits": ["train", "val"],
            "official_test_pixels_authorized": False,
            "action_program_json_authorized": False,
            "proposer_or_model_authorized": False,
            "candidate_dependent_extraction_authorized": False,
        },
    }
    input_digest = _address(input_commitment)
    selected_ids = tuple(item.task_id for item in selection.selected)
    groups = split_index.canonical_groups
    successor = predecessor.record(
        phase="relational-coverage-drill",
        actor=actor,
        purpose=(
            "candidate-independent Python geometry/contact coverage on a bounded "
            "exact-unused train/validation pilot"
        ),
        task_ids=selected_ids,
        source=f"relational-coverage-input:{input_digest}",
        observed_at=observed_at,
        known_task_ids=set(groups["train"]) | set(groups["val"]) | set(groups["test"]),
        sealed_task_ids=groups["test"],
        require_unseen=True,
    )
    # This persistence and cold reload are intentionally before even resolving
    # a selected task directory.  Tests inject a reader that asserts this edge.
    successor_path = _persist_successor(successor, Path(exposure_store))

    aggregates = _AggregateBook()
    selected_tasks_manifest: list[dict[str, object]] = []
    panel_receipts: list[dict[str, object]] = []
    for item in selection.selected:
        aggregates.record_task(item)
        task_panels: list[dict[str, object]] = []
        paths = _task_panel_paths(root, item)
        if len(paths) != 14:
            raise CoverageDrillError("selected task does not name exactly 14 panels")
        for polarity, index, path in paths:
            payload = png_reader(path)
            if not isinstance(payload, bytes) or not payload.startswith(PNG_SIGNATURE):
                raise CoverageDrillError("PNG reader returned non-PNG bytes")
            label = "1" if polarity == "positive" else "0"
            panel_id = f"{item.family}/{item.task_id}/{label}/{index}.png"
            panel_address = _bytes_address(payload)
            task_panels.append(
                {
                    "panel_id": panel_id,
                    "polarity": polarity,
                    "index": index,
                    "filename": f"{index}.png",
                    "sha256": panel_address,
                    "size_bytes": len(payload),
                }
            )
            try:
                packet = extractor(payload)
            except Exception as exc:  # one panel failure must not become absence
                aggregates.record_error(item, polarity, exc)
                panel_receipts.append(
                    {
                        "panel_id": panel_id,
                        "png_sha256": panel_address,
                        "status": "error",
                        "error_type": type(exc).__module__ + "." + type(exc).__qualname__,
                        "loop_scene_packet_digest": None,
                    }
                )
            else:
                if not isinstance(packet, LoopScenePacket):
                    raise CoverageDrillError(
                        "extractor violated the LoopScenePacket return contract"
                    )
                if packet.panel_digest != panel_address.removeprefix("sha256:"):
                    raise CoverageDrillError(
                        "loop scene packet is not bound to selected PNG bytes"
                    )
                _assert_packet_identity(packet, algorithms)
                aggregates.record_packet(item, polarity, packet)
                panel_receipts.append(
                    {
                        "panel_id": panel_id,
                        "png_sha256": panel_address,
                        "status": "present",
                        "error_type": None,
                        "loop_scene_packet_digest": packet.digest(),
                    }
                )
        task_content = {
            "task_id": item.task_id,
            "family": item.family,
            "split": item.split,
            "generator": item.generator,
            "panels": task_panels,
        }
        selected_tasks_manifest.append(
            {**task_content, "digest": _address(task_content)}
        )

    manifest_content = {
        "schema": SELECTED_MANIFEST_SCHEMA,
        "source_corpus_manifest_digest": source_corpus_manifest_digest,
        "split_source_digest": split_index.source_digest,
        "selection_digest": selection.digest,
        "tasks": selected_tasks_manifest,
    }
    selected_manifest = {
        **manifest_content,
        "digest": _address(manifest_content),
    }
    report_content: dict[str, object] = {
        "schema": SCHEMA,
        "algorithm_id": ALGORITHM_ID,
        "input_digest": input_digest,
        "source": {
            "corpus_manifest_digest": source_corpus_manifest_digest,
            "split_source_digest": split_index.source_digest,
        },
        "exposure": {
            "predecessor_digest": predecessor.digest,
            "successor_digest": successor.digest,
            "successor_event_count": len(successor.events),
            "successor_filename": successor_path.name,
            "precommit_before_selected_png_access": True,
        },
        "restrictions": input_commitment["restrictions"],
        "algorithm_identities": algorithms,
        "selection": selection.to_data(),
        "selected_task_manifest": selected_manifest,
        "panel_receipts": sorted(panel_receipts, key=lambda value: value["panel_id"]),
        "aggregates": aggregates.to_data(),
    }
    report: dict[str, object] = {
        **report_content,
        "output_digest": _address(report_content),
    }
    output_path = _write_once_durable(
        Path(output_store)
        / (report["output_digest"].removeprefix("sha256:") + ".coverage.json"),
        canonical_json(report) + b"\n",
    )
    return CoverageDrillResult(
        report=report,
        report_path=output_path,
        exposure_successor=successor,
        exposure_successor_path=successor_path,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run an offline train/validation relational witness coverage drill."
    )
    parser.add_argument("--corpus-root", required=True, type=Path)
    parser.add_argument("--split-file", required=True, type=Path)
    parser.add_argument("--corpus-manifest-digest", required=True)
    parser.add_argument("--ledger-in", required=True, type=Path)
    parser.add_argument("--ledger-store", required=True, type=Path)
    parser.add_argument("--output-store", required=True, type=Path)
    parser.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    parser.add_argument("--per-generator", type=int, default=1)
    parser.add_argument("--per-split-family", type=int, default=12)
    parser.add_argument(
        "--minimum-strict-dev-reserve",
        type=int,
        default=STRICT_DEV_CAPACITY_REFERENCE,
        help="default 16; use 0 only for synthetic fixtures, never routine drilling",
    )
    parser.add_argument("--actor", default="offline-relational-coverage-drill")
    parser.add_argument("--observed-at")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_coverage_drill(
        corpus_root=args.corpus_root,
        split_index=SplitIndex.load(args.split_file),
        source_corpus_manifest_digest=args.corpus_manifest_digest,
        predecessor=ExposureLedger.load(args.ledger_in),
        exposure_store=args.ledger_store,
        output_store=args.output_store,
        namespace=args.namespace,
        per_generator=args.per_generator,
        per_split_family=args.per_split_family,
        minimum_strict_dev_reserve=args.minimum_strict_dev_reserve,
        require_official_split=True,
        actor=args.actor,
        observed_at=args.observed_at,
    )
    print(
        json.dumps(
            {
                "output_digest": result.report["output_digest"],
                "report_path": str(result.report_path),
                "exposure_successor_digest": result.exposure_successor.digest,
                "exposure_successor_path": str(result.exposure_successor_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised as a module CLI
    raise SystemExit(main())


__all__ = [
    "ALGORITHM_ID",
    "CoverageDrillError",
    "CoverageDrillResult",
    "CoverageSelection",
    "DEFAULT_NAMESPACE",
    "SCHEMA",
    "STRICT_DEV_CAPACITY_REFERENCE",
    "SelectedTask",
    "main",
    "run_coverage_drill",
    "select_exact_unused_pilot",
    "task_generator_key",
]

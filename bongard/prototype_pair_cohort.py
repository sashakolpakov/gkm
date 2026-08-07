"""Metadata-only planning for prototype-conditioned Basic pair drills.

This module deliberately accepts no corpus tree, panel manifest, panel path, or
action-program value.  The only release data it consumes are the authenticated
split JSON bytes and the release's sorted task-ID inventory.  It plans a
targeted engineering exercise in which two opaque visual tags are grounded by
exact generator-shape prototypes and cross-calibrated on other Basic pairs:

* an ``A + other`` positive panel is weakly labelled ``A present, B absent``;
* a ``B + other`` positive panel is weakly labelled ``B present, A absent``.

The labels concern exact names passed to the pinned Basic generator.  They do
not prove an arbitrary prose description, perceptual equivalence, or the
absence of a visual lookalike.  Prototype and calibration disclosure also
makes the selected drill semantically reused, even though every selected
official task identity must be exact-unused at planning time.  Consequently
the artifact authorizes a targeted engineering claim only, never a benchmark,
unseen-data, validation, or official-test claim.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from bongard.canonical import canonical_digest
from bongard.cluster_binomial import familywise_clopper_pearson_upper_ppm
from bongard.corpus import CorpusError, SplitIndex
from bongard.exposure import (
    ExposureLedger,
    semantic_resolver_policy_digest,
    task_id_from_panel_id,
)
from bongard.grounded_multimodal_predicates import PYTHON_PREDICATE_AUTHORITY_ID
from bongard.historical_exposure import HistoricalExposureSeed
from bongard.release import OfficialReleaseDescriptor


PLAN_SCHEMA = "gkm.bongard-prototype-pair-cohort-plan.v1"
PROTOTYPE_SCHEMA = "gkm.bongard-prototype-pair-binding.v1"
CALIBRATION_SCHEMA = "gkm.bongard-prototype-pair-calibration-cluster.v1"
DRILL_SCHEMA = "gkm.bongard-prototype-pair-drill.v1"
CANDIDATE_SCHEMA = "gkm.bongard-prototype-pair-candidate.v1"
SEED_COMMITMENT_SCHEMA = "gkm.bongard-prototype-pair-seed-commitment.v1"
ALGORITHM_ID = "bongard.prototype-pair/hash-ranked-cross-calibration-14-v1"
DEFAULT_NAMESPACE = "bongard-prototype-pair-bd-train-v1"

MIN_CALIBRATION_CLUSTERS = 14
CALIBRATION_CLUSTERS_PER_TAG = 14
PROTOTYPE_POSITIVE_INDICES = (0, 3, 6)
DRILL_POSITIVE_INDICES = tuple(range(7))
DRILL_NEGATIVE_INDICES = tuple(range(7))
OPAQUE_TAG_IDS = ("opaque_visual_tag_0", "opaque_visual_tag_1")
BIRD_FAMILIES = tuple(f"bird{index}" for index in range(1, 9))

HYPOTHESIS_COUNT = 4
CONFIDENCE_LEVEL_PPM = 950_000
TARGETED_ENGINEERING_TOLERANCE_PPM = 300_000
ZERO_ERROR_FAMILY_UPPER_PPM = 268_752

PYTHON_AUTHORITY_ID = PYTHON_PREDICATE_AUTHORITY_ID
OFFICIAL_UPSTREAM_REPOSITORY = "https://github.com/NVlabs/Bongard-LOGO"
OFFICIAL_UPSTREAM_COMMIT = "9df7c78ee9c6a2ff041b48d9ed407359aac259c3"
OFFICIAL_BASIC_SAMPLER_SHA256 = (
    "c43c04d161b5a46ee5a319b9b618bb2d6618f85db64defd42b10ddd03baad537"
)
OFFICIAL_BASIC_GENERATOR_SHA256 = (
    "b54dcc9b9d06b8e30e223a2653669d1eae3ba17a950e3d093f832916b8e45606"
)

WEAK_LABEL_AUTHORITY = (
    "Pinned Basic generator identity only: a selected positive panel names "
    "exactly the two ordered task shapes.  This weak label grounds opaque "
    "prototype-conditioned tags; it does not prove arbitrary prose semantics "
    "or exclude perceptual lookalikes."
)
ENGINEERING_CLAIM = (
    "Exact-unused TRAIN task identities used after prototype and calibration "
    "semantic disclosure; targeted engineering only, not an unseen benchmark."
)

_ADDRESS = re.compile(r"sha256:[0-9a-f]{64}\Z")
_RAW_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_BASIC_TASK = re.compile(r"bd_(.+)_0000\Z")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/-]{0,255}\Z")
_MAX_SPLIT_BYTES = 16 * 1024 * 1024

# This is computed from the imported module bytes, not copied by hand.  A cold
# process therefore gives modified planner code a different authority even if
# an attacker preserves every schema string and policy constant below.
PLANNER_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def planner_algorithm_digest() -> str:
    """Bind the exact Python planner bytes and every selection/claim constant."""

    return _address(
        {
            "schema": "gkm.bongard-prototype-pair-planner-algorithm.v1",
            "planner_source_sha256": PLANNER_SOURCE_SHA256,
            "algorithm_id": ALGORITHM_ID,
            "plan_schema": PLAN_SCHEMA,
            "candidate_rule": {
                "split": "train",
                "exact_task_exposure": "unused",
                "pair_arity": 2,
                "prototype_singletons_exact_unused": True,
                "unique_joint_task": True,
                "minimum_clean_other_pairs_per_shape": MIN_CALIBRATION_CLUSTERS,
            },
            "selection": "external-seed-content-hash-rank",
            "opaque_tag_ids": list(OPAQUE_TAG_IDS),
            "prototype_positive_indices": list(PROTOTYPE_POSITIVE_INDICES),
            "calibration_clusters_per_tag": CALIBRATION_CLUSTERS_PER_TAG,
            "calibration_panel_side": "positive",
            "calibration_panel_index_domain": list(range(7)),
            "calibration_labeling": "cross-present-absent-score-both-tags",
            "drill_positive_indices": list(DRILL_POSITIVE_INDICES),
            "drill_negative_indices": list(DRILL_NEGATIVE_INDICES),
            "hypothesis_count": HYPOTHESIS_COUNT,
            "confidence_level_ppm": CONFIDENCE_LEVEL_PPM,
            "targeted_engineering_tolerance_ppm": (
                TARGETED_ENGINEERING_TOLERANCE_PPM
            ),
            "zero_error_family_upper_ppm": ZERO_ERROR_FAMILY_UPPER_PPM,
            "weak_label_authority": WEAK_LABEL_AUTHORITY,
            "engineering_claim": ENGINEERING_CLAIM,
            "basic_sampler_sha256": OFFICIAL_BASIC_SAMPLER_SHA256,
            "basic_generator_sha256": OFFICIAL_BASIC_GENERATOR_SHA256,
            "predicate_authority_id": PYTHON_AUTHORITY_ID,
            "python_is_canonical_authority": True,
            "lean_required": False,
            "lean_defines_artifact_identity": False,
            "lean_affects_selection_or_decision": False,
            "lean_required_for_replay": False,
        }
    )


class PrototypePairCohortError(ValueError):
    """An input authority, selection, or replay invariant failed."""


def _address(value: object) -> str:
    return "sha256:" + canonical_digest(value)


def _require_address(value: object, label: str) -> str:
    if not isinstance(value, str) or _ADDRESS.fullmatch(value) is None:
        raise PrototypePairCohortError(f"{label} must be a sha256: address")
    return value


def _require_raw_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or _RAW_SHA256.fullmatch(value) is None:
        raise PrototypePairCohortError(f"{label} must be lowercase SHA-256")
    return value


def _require_identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise PrototypePairCohortError(f"{label} must be a bounded identifier")
    return value


def _strict_object(
    value: object, fields: set[str], label: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise PrototypePairCohortError(f"{label} must be an object")
    if set(value) != fields:
        raise PrototypePairCohortError(
            f"{label} fields differ: missing={sorted(fields - set(value))}, "
            f"extra={sorted(set(value) - fields)}"
        )
    return value


def _strict_list(value: object, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise PrototypePairCohortError(f"{label} must be a list")
    return value


def _integer(
    value: object,
    label: str,
    *,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
        or (maximum is not None and value > maximum)
    ):
        raise PrototypePairCohortError(f"{label} is outside its integer domain")
    return value


def _verify_serialized_digest(
    value: Mapping[str, Any], *, label: str
) -> None:
    digest = _require_address(value["record_digest"], f"{label} record digest")
    body = {key: item for key, item in value.items() if key != "record_digest"}
    if digest != _address(body):
        raise PrototypePairCohortError(f"{label} record digest differs")


def _selection_seed(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or "\x00" in value
        or len(value.encode("utf-8", errors="strict")) > 4096
    ):
        raise PrototypePairCohortError(
            "selection seed must be bounded, stripped, and NUL-free"
        )
    return value


def _seed_digest(seed: str) -> str:
    return "sha256:" + hashlib.sha256(seed.encode("utf-8")).hexdigest()


def task_id_inventory_digest(task_ids: Sequence[str]) -> str:
    """Return the official release's sorted-line task inventory address."""

    values = tuple(task_ids)
    if (
        not values
        or any(not isinstance(task_id, str) or not task_id for task_id in values)
        or values != tuple(sorted(set(values)))
    ):
        raise PrototypePairCohortError(
            "task-ID inventory must be nonempty, unique, and sorted"
        )
    payload = "".join(f"{task_id}\n" for task_id in values).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def prototype_pair_seed_commitment(
    selection_seed: str, *, namespace: str = DEFAULT_NAMESPACE
) -> str:
    """Commit an external seed before any candidate or panel is selected."""

    seed = _selection_seed(selection_seed)
    _require_identifier(namespace, "namespace")
    return _address(
        {
            "schema": SEED_COMMITMENT_SCHEMA,
            "algorithm_id": ALGORITHM_ID,
            "namespace": namespace,
            "selection_seed_digest": _seed_digest(seed),
        }
    )


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise PrototypePairCohortError(f"split JSON repeats key {key!r}")
        result[key] = value
    return result


def _authenticated_split(
    split_bytes: bytes,
    *,
    expected_digest: str,
    expected_size: int,
) -> SplitIndex:
    if not isinstance(split_bytes, bytes):
        raise TypeError("split_bytes must be exact bytes")
    if not split_bytes or len(split_bytes) > _MAX_SPLIT_BYTES:
        raise PrototypePairCohortError("split bytes are empty or exceed the cap")
    if isinstance(expected_size, bool) or not isinstance(expected_size, int):
        raise PrototypePairCohortError("split size pin must be an integer")
    actual = "sha256:" + hashlib.sha256(split_bytes).hexdigest()
    if len(split_bytes) != expected_size or actual != _require_address(
        expected_digest, "split source digest"
    ):
        raise PrototypePairCohortError("split bytes differ from release authority")
    try:
        raw = json.loads(
            split_bytes.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise PrototypePairCohortError(f"split bytes are not strict JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise PrototypePairCohortError("split JSON root must be an object")
    groups: list[tuple[str, tuple[str, ...]]] = []
    for name, members in raw.items():
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(members, list)
            or any(not isinstance(task_id, str) or not task_id for task_id in members)
            or len(members) != len(set(members))
        ):
            raise PrototypePairCohortError("split JSON contains an invalid group")
        groups.append((name, tuple(sorted(members))))
    return SplitIndex(
        groups=tuple(sorted(groups)),
        source_path=None,
        source_digest=actual,
    )


def _parse_basic_task_id(
    task_id: str, vocabulary: frozenset[str]
) -> tuple[str, ...]:
    """Parse a singleton or ordered pair against the frozen Basic vocabulary."""

    if not isinstance(task_id, str):
        raise PrototypePairCohortError("Basic task ID must be text")
    match = _BASIC_TASK.fullmatch(task_id)
    if match is None:
        raise PrototypePairCohortError(f"malformed Basic task ID: {task_id!r}")
    body = match.group(1)
    parses: list[tuple[str, ...]] = []
    if body in vocabulary:
        parses.append((body,))
    for index, character in enumerate(body):
        if character != "-":
            continue
        left, right = body[:index], body[index + 1 :]
        if left in vocabulary and right in vocabulary and left != right:
            parses.append((left, right))
    if len(parses) != 1:
        qualifier = "unknown" if not parses else "ambiguous"
        raise PrototypePairCohortError(
            f"{qualifier} Basic shape expression in {task_id!r}"
        )
    return parses[0]


def _panel_id(task_id: str, side: str, index: int) -> str:
    if side not in {"positive", "negative"} or not 0 <= index < 7:
        raise PrototypePairCohortError("panel schedule is outside official 7+7")
    label = "1" if side == "positive" else "0"
    return f"bd/{task_id}/{label}/{index}.png"


def _rank(
    *,
    namespace: str,
    seed_digest: str,
    role: str,
    task_id: str,
    tag_id: str | None = None,
) -> str:
    return canonical_digest(
        {
            "algorithm_id": ALGORITHM_ID,
            "namespace": namespace,
            "selection_seed_digest": seed_digest,
            "role": role,
            "tag_id": tag_id,
            "task_id": task_id,
        }
    )


def _calibration_index(
    *, namespace: str, seed_digest: str, tag_id: str, task_id: str
) -> int:
    digest = canonical_digest(
        {
            "algorithm_id": ALGORITHM_ID,
            "namespace": namespace,
            "selection_seed_digest": seed_digest,
            "role": "positive-side-calibration-panel-index",
            "tag_id": tag_id,
            "task_id": task_id,
            "index_domain": list(range(7)),
        }
    )
    return int(digest, 16) % 7


@dataclass(frozen=True, slots=True)
class PrototypeBinding:
    tag_id: str
    shape_family: str
    task_id: str
    cluster_id: str
    side: str
    panel_indices: tuple[int, ...]
    panel_ids: tuple[str, ...]
    exact_task_unused: bool
    weak_generator_identity_label: bool

    def __post_init__(self) -> None:
        _require_identifier(self.tag_id, "prototype tag")
        _require_identifier(self.shape_family, "prototype shape")
        if (
            self.task_id != f"bd_{self.shape_family}_0000"
            or self.cluster_id != self.task_id
            or self.side != "positive"
            or self.panel_indices != PROTOTYPE_POSITIVE_INDICES
            or self.panel_ids
            != tuple(_panel_id(self.task_id, self.side, i) for i in self.panel_indices)
            or self.exact_task_unused is not True
            or self.weak_generator_identity_label is not True
        ):
            raise PrototypePairCohortError("prototype binding differs from policy")

    def to_data(self) -> dict[str, object]:
        body: dict[str, object] = {
            "schema": PROTOTYPE_SCHEMA,
            "tag_id": self.tag_id,
            "shape_family": self.shape_family,
            "task_id": self.task_id,
            "cluster_id": self.cluster_id,
            "side": self.side,
            "panel_indices": list(self.panel_indices),
            "panel_ids": list(self.panel_ids),
            "exact_task_unused": self.exact_task_unused,
            "weak_generator_identity_label": self.weak_generator_identity_label,
        }
        return {**body, "record_digest": _address(body)}

    @classmethod
    def from_data(cls, value: Mapping[str, Any]) -> "PrototypeBinding":
        fields = {
            "schema",
            "tag_id",
            "shape_family",
            "task_id",
            "cluster_id",
            "side",
            "panel_indices",
            "panel_ids",
            "exact_task_unused",
            "weak_generator_identity_label",
            "record_digest",
        }
        raw = _strict_object(value, fields, "prototype binding")
        _verify_serialized_digest(raw, label="prototype binding")
        if raw["schema"] != PROTOTYPE_SCHEMA:
            raise PrototypePairCohortError("prototype binding schema differs")
        result = cls(
            tag_id=raw["tag_id"],
            shape_family=raw["shape_family"],
            task_id=raw["task_id"],
            cluster_id=raw["cluster_id"],
            side=raw["side"],
            panel_indices=tuple(
                _strict_list(raw["panel_indices"], "prototype panel indices")
            ),
            panel_ids=tuple(
                _strict_list(raw["panel_ids"], "prototype panel IDs")
            ),
            exact_task_unused=raw["exact_task_unused"],
            weak_generator_identity_label=raw[
                "weak_generator_identity_label"
            ],
        )
        if result.to_data() != dict(raw):
            raise PrototypePairCohortError("prototype binding is not canonical")
        return result


@dataclass(frozen=True, slots=True)
class CalibrationCluster:
    task_id: str
    ordered_shapes: tuple[str, str]
    cluster_id: str
    side: str
    panel_index: int
    panel_id: str
    score_tag_ids: tuple[str, str]
    expected_tag_states: tuple[tuple[str, str], tuple[str, str]]
    group_tag_id: str
    selection_rank: str
    exact_task_unused: bool

    def __post_init__(self) -> None:
        if (
            len(self.expected_tag_states) != 2
            or any(
                not isinstance(row, tuple) or len(row) != 2
                for row in self.expected_tag_states
            )
        ):
            raise PrototypePairCohortError("expected tag-state rows differ")
        if (
            len(self.ordered_shapes) != 2
            or self.ordered_shapes[0] == self.ordered_shapes[1]
            or any(not isinstance(value, str) or not value for value in self.ordered_shapes)
            or self.task_id
            != f"bd_{self.ordered_shapes[0]}-{self.ordered_shapes[1]}_0000"
            or self.cluster_id != self.task_id
            or self.side != "positive"
            or isinstance(self.panel_index, bool)
            or not isinstance(self.panel_index, int)
            or not 0 <= self.panel_index < 7
            or self.panel_id != _panel_id(self.task_id, self.side, self.panel_index)
            or self.score_tag_ids != OPAQUE_TAG_IDS
            or self.group_tag_id not in OPAQUE_TAG_IDS
            or any(
                not isinstance(tag_id, str) or not isinstance(state, str)
                for tag_id, state in self.expected_tag_states
            )
            or tuple(tag_id for tag_id, _state in self.expected_tag_states)
            != OPAQUE_TAG_IDS
            or sorted(state for _tag_id, state in self.expected_tag_states)
            != ["absent", "present"]
            or dict(self.expected_tag_states)[self.group_tag_id] != "present"
            or not isinstance(self.selection_rank, str)
            or _RAW_SHA256.fullmatch(self.selection_rank) is None
            or self.exact_task_unused is not True
        ):
            raise PrototypePairCohortError("calibration cluster differs from policy")

    def to_data(self) -> dict[str, object]:
        body: dict[str, object] = {
            "schema": CALIBRATION_SCHEMA,
            "task_id": self.task_id,
            "ordered_shapes": list(self.ordered_shapes),
            "cluster_id": self.cluster_id,
            "side": self.side,
            "panel_index": self.panel_index,
            "panel_id": self.panel_id,
            "score_tag_ids": list(self.score_tag_ids),
            "expected_tag_states": [
                {"tag_id": tag_id, "state": state}
                for tag_id, state in self.expected_tag_states
            ],
            "group_tag_id": self.group_tag_id,
            "selection_rank": self.selection_rank,
            "exact_task_unused": self.exact_task_unused,
        }
        return {**body, "record_digest": _address(body)}

    @classmethod
    def from_data(cls, value: Mapping[str, Any]) -> "CalibrationCluster":
        fields = {
            "schema",
            "task_id",
            "ordered_shapes",
            "cluster_id",
            "side",
            "panel_index",
            "panel_id",
            "score_tag_ids",
            "expected_tag_states",
            "group_tag_id",
            "selection_rank",
            "exact_task_unused",
            "record_digest",
        }
        raw = _strict_object(value, fields, "calibration cluster")
        _verify_serialized_digest(raw, label="calibration cluster")
        if raw["schema"] != CALIBRATION_SCHEMA:
            raise PrototypePairCohortError("calibration cluster schema differs")
        state_rows = _strict_list(
            raw["expected_tag_states"], "expected tag states"
        )
        states: list[tuple[str, str]] = []
        for row in state_rows:
            item = _strict_object(
                row, {"tag_id", "state"}, "expected tag state"
            )
            states.append((item["tag_id"], item["state"]))
        result = cls(
            task_id=raw["task_id"],
            ordered_shapes=tuple(
                _strict_list(raw["ordered_shapes"], "ordered shapes")
            ),  # type: ignore[arg-type]
            cluster_id=raw["cluster_id"],
            side=raw["side"],
            panel_index=raw["panel_index"],
            panel_id=raw["panel_id"],
            score_tag_ids=tuple(
                _strict_list(raw["score_tag_ids"], "score tag IDs")
            ),  # type: ignore[arg-type]
            expected_tag_states=tuple(states),  # type: ignore[arg-type]
            group_tag_id=raw["group_tag_id"],
            selection_rank=raw["selection_rank"],
            exact_task_unused=raw["exact_task_unused"],
        )
        if result.to_data() != dict(raw):
            raise PrototypePairCohortError("calibration cluster is not canonical")
        return result


@dataclass(frozen=True, slots=True)
class DrillSchedule:
    task_id: str
    ordered_shapes: tuple[str, str]
    cluster_id: str
    selection_rank: str
    positive_indices: tuple[int, ...]
    negative_indices: tuple[int, ...]
    positive_panel_ids: tuple[str, ...]
    negative_panel_ids: tuple[str, ...]
    exact_task_unused: bool
    pixels_opened_during_planning: bool

    def __post_init__(self) -> None:
        if (
            len(self.ordered_shapes) != 2
            or any(not isinstance(shape, str) or not shape for shape in self.ordered_shapes)
            or self.ordered_shapes[0] == self.ordered_shapes[1]
            or self.task_id
            != f"bd_{self.ordered_shapes[0]}-{self.ordered_shapes[1]}_0000"
            or self.cluster_id != self.task_id
            or not isinstance(self.selection_rank, str)
            or _RAW_SHA256.fullmatch(self.selection_rank) is None
            or self.positive_indices != DRILL_POSITIVE_INDICES
            or self.negative_indices != DRILL_NEGATIVE_INDICES
            or self.positive_panel_ids
            != tuple(_panel_id(self.task_id, "positive", i) for i in self.positive_indices)
            or self.negative_panel_ids
            != tuple(_panel_id(self.task_id, "negative", i) for i in self.negative_indices)
            or self.exact_task_unused is not True
            or self.pixels_opened_during_planning is not False
        ):
            raise PrototypePairCohortError("drill schedule differs from policy")

    def to_data(self) -> dict[str, object]:
        body: dict[str, object] = {
            "schema": DRILL_SCHEMA,
            "task_id": self.task_id,
            "ordered_shapes": list(self.ordered_shapes),
            "cluster_id": self.cluster_id,
            "selection_rank": self.selection_rank,
            "positive_indices": list(self.positive_indices),
            "negative_indices": list(self.negative_indices),
            "positive_panel_ids": list(self.positive_panel_ids),
            "negative_panel_ids": list(self.negative_panel_ids),
            "exact_task_unused": self.exact_task_unused,
            "pixels_opened_during_planning": self.pixels_opened_during_planning,
        }
        return {**body, "record_digest": _address(body)}

    @classmethod
    def from_data(cls, value: Mapping[str, Any]) -> "DrillSchedule":
        fields = {
            "schema",
            "task_id",
            "ordered_shapes",
            "cluster_id",
            "selection_rank",
            "positive_indices",
            "negative_indices",
            "positive_panel_ids",
            "negative_panel_ids",
            "exact_task_unused",
            "pixels_opened_during_planning",
            "record_digest",
        }
        raw = _strict_object(value, fields, "drill schedule")
        _verify_serialized_digest(raw, label="drill schedule")
        if raw["schema"] != DRILL_SCHEMA:
            raise PrototypePairCohortError("drill schedule schema differs")
        result = cls(
            task_id=raw["task_id"],
            ordered_shapes=tuple(
                _strict_list(raw["ordered_shapes"], "drill ordered shapes")
            ),  # type: ignore[arg-type]
            cluster_id=raw["cluster_id"],
            selection_rank=raw["selection_rank"],
            positive_indices=tuple(
                _strict_list(raw["positive_indices"], "positive indices")
            ),
            negative_indices=tuple(
                _strict_list(raw["negative_indices"], "negative indices")
            ),
            positive_panel_ids=tuple(
                _strict_list(raw["positive_panel_ids"], "positive panel IDs")
            ),
            negative_panel_ids=tuple(
                _strict_list(raw["negative_panel_ids"], "negative panel IDs")
            ),
            exact_task_unused=raw["exact_task_unused"],
            pixels_opened_during_planning=raw[
                "pixels_opened_during_planning"
            ],
        )
        if result.to_data() != dict(raw):
            raise PrototypePairCohortError("drill schedule is not canonical")
        return result


@dataclass(frozen=True, slots=True)
class CandidateSummary:
    task_id: str
    ordered_shapes: tuple[str, str]
    first_shape_other_cluster_count: int
    second_shape_other_cluster_count: int
    unique_joint_task: bool
    both_prototypes_exact_unused: bool
    bird_family_matches: tuple[str, ...]

    def __post_init__(self) -> None:
        _integer(
            self.first_shape_other_cluster_count,
            "first shape other-cluster count",
            minimum=MIN_CALIBRATION_CLUSTERS,
        )
        _integer(
            self.second_shape_other_cluster_count,
            "second shape other-cluster count",
            minimum=MIN_CALIBRATION_CLUSTERS,
        )
        if (
            len(self.ordered_shapes) != 2
            or any(not isinstance(shape, str) or not shape for shape in self.ordered_shapes)
            or self.task_id
            != f"bd_{self.ordered_shapes[0]}-{self.ordered_shapes[1]}_0000"
            or self.unique_joint_task is not True
            or self.both_prototypes_exact_unused is not True
            or self.bird_family_matches
            != tuple(shape for shape in self.ordered_shapes if shape in BIRD_FAMILIES)
        ):
            raise PrototypePairCohortError("candidate summary differs from policy")

    def to_data(self) -> dict[str, object]:
        body: dict[str, object] = {
            "schema": CANDIDATE_SCHEMA,
            "task_id": self.task_id,
            "ordered_shapes": list(self.ordered_shapes),
            "other_cluster_counts": {
                self.ordered_shapes[0]: self.first_shape_other_cluster_count,
                self.ordered_shapes[1]: self.second_shape_other_cluster_count,
            },
            "unique_joint_task": self.unique_joint_task,
            "both_prototypes_exact_unused": self.both_prototypes_exact_unused,
            "bird_family_matches": list(self.bird_family_matches),
        }
        return {**body, "record_digest": _address(body)}

    @classmethod
    def from_data(cls, value: Mapping[str, Any]) -> "CandidateSummary":
        fields = {
            "schema",
            "task_id",
            "ordered_shapes",
            "other_cluster_counts",
            "unique_joint_task",
            "both_prototypes_exact_unused",
            "bird_family_matches",
            "record_digest",
        }
        raw = _strict_object(value, fields, "candidate summary")
        _verify_serialized_digest(raw, label="candidate summary")
        if raw["schema"] != CANDIDATE_SCHEMA:
            raise PrototypePairCohortError("candidate schema differs")
        shapes = tuple(
            _strict_list(raw["ordered_shapes"], "candidate ordered shapes")
        )
        if (
            len(shapes) != 2
            or any(not isinstance(shape, str) or not shape for shape in shapes)
            or shapes[0] == shapes[1]
        ):
            raise PrototypePairCohortError("candidate must name two shapes")
        counts = _strict_object(
            raw["other_cluster_counts"],
            {shapes[0], shapes[1]},
            "candidate other-cluster counts",
        )
        result = cls(
            task_id=raw["task_id"],
            ordered_shapes=shapes,  # type: ignore[arg-type]
            first_shape_other_cluster_count=counts[shapes[0]],
            second_shape_other_cluster_count=counts[shapes[1]],
            unique_joint_task=raw["unique_joint_task"],
            both_prototypes_exact_unused=raw[
                "both_prototypes_exact_unused"
            ],
            bird_family_matches=tuple(
                _strict_list(raw["bird_family_matches"], "bird family matches")
            ),
        )
        if result.to_data() != dict(raw):
            raise PrototypePairCohortError("candidate summary is not canonical")
        return result


@dataclass(frozen=True, slots=True)
class PrototypePairCohortPlan:
    namespace: str
    selection_seed_digest: str
    selection_seed_commitment: str
    release_descriptor_digest: str
    release_id: str
    archive_sha256: str
    corpus_manifest_digest: str
    split_source_digest: str
    split_metadata_digest: str
    task_inventory_digest: str
    task_inventory_count: int
    historical_seed_digest: str
    resolver_policy_digest: str
    exposure_predecessor_digest: str
    active_semantic_resolution_digest: str
    upstream_repository: str
    upstream_commit: str
    basic_sampler_sha256: str
    basic_generator_sha256: str
    candidates: tuple[CandidateSummary, ...]
    bird_candidate_task_ids: tuple[str, ...]
    excluded_exact_used_train_basic_task_ids: tuple[str, ...]
    prototypes: tuple[PrototypeBinding, PrototypeBinding]
    calibration_clusters: tuple[CalibrationCluster, ...]
    drill: DrillSchedule
    hypothesis_count: int
    clusters_per_hypothesis: int
    confidence_level_ppm: int
    zero_error_family_upper_ppm: int
    targeted_engineering_tolerance_ppm: int
    zero_errors_required_for_tolerance: bool
    stronger_250k_claim_authorized: bool
    thresholds_must_be_frozen_before_calibration: bool
    weak_label_authority: str
    engineering_claim: str
    drill_semantics_reused: bool
    benchmark_claim_authorized: bool
    unseen_claim_authorized: bool
    validation_split_authorized: bool
    official_test_authorized: bool
    panel_bytes_read: bool
    panel_paths_resolved: bool
    action_program_json_authorized: bool
    action_program_json_read: bool
    planner_source_sha256: str
    planner_algorithm_digest: str
    predicate_authority_id: str
    python_is_canonical_authority: bool
    lean_required: bool
    lean_defines_artifact_identity: bool
    lean_affects_selection_or_decision: bool
    lean_required_for_replay: bool
    optional_secondary_checker_detachable: bool
    algorithm_id: str

    def __post_init__(self) -> None:
        _require_identifier(self.namespace, "namespace")
        if not isinstance(self.release_id, str) or not self.release_id:
            raise PrototypePairCohortError("release ID must be nonempty text")
        _integer(self.task_inventory_count, "task inventory count", minimum=1)
        for name in (
            "selection_seed_digest",
            "selection_seed_commitment",
            "release_descriptor_digest",
            "archive_sha256",
            "corpus_manifest_digest",
            "split_source_digest",
            "split_metadata_digest",
            "task_inventory_digest",
            "historical_seed_digest",
            "resolver_policy_digest",
            "exposure_predecessor_digest",
            "active_semantic_resolution_digest",
        ):
            _require_address(getattr(self, name), name)
        _require_raw_sha256(self.basic_sampler_sha256, "basic sampler SHA-256")
        _require_raw_sha256(self.basic_generator_sha256, "basic generator SHA-256")
        _require_raw_sha256(self.planner_source_sha256, "planner source SHA-256")
        _require_address(self.planner_algorithm_digest, "planner algorithm digest")
        if (
            not isinstance(self.upstream_commit, str)
            or _COMMIT.fullmatch(self.upstream_commit) is None
        ):
            raise PrototypePairCohortError("upstream commit must be exact 40-hex")
        if (
            not self.candidates
            or any(not isinstance(item, CandidateSummary) for item in self.candidates)
            or tuple(item.task_id for item in self.candidates)
            != tuple(sorted(item.task_id for item in self.candidates))
        ):
            raise PrototypePairCohortError("candidate inventory must be nonempty and sorted")
        expected_birds = tuple(
            item.task_id for item in self.candidates if item.bird_family_matches
        )
        if (
            any(
                not isinstance(task_id, str) or not task_id
                for task_id in self.bird_candidate_task_ids
            )
            or self.bird_candidate_task_ids != expected_birds
        ):
            raise PrototypePairCohortError("bird candidate report differs")
        if (
            any(
                not isinstance(task_id, str) or not task_id
                for task_id in self.excluded_exact_used_train_basic_task_ids
            )
            or self.excluded_exact_used_train_basic_task_ids
            != tuple(sorted(set(self.excluded_exact_used_train_basic_task_ids)))
        ):
            raise PrototypePairCohortError("exact-used exclusion list differs")
        if (
            len(self.prototypes) != 2
            or any(not isinstance(item, PrototypeBinding) for item in self.prototypes)
            or not isinstance(self.drill, DrillSchedule)
            or tuple(item.tag_id for item in self.prototypes) != OPAQUE_TAG_IDS
            or tuple(item.shape_family for item in self.prototypes)
            != self.drill.ordered_shapes
        ):
            raise PrototypePairCohortError("prototype/tag mapping differs")
        if (
            len(self.calibration_clusters) != 2 * CALIBRATION_CLUSTERS_PER_TAG
            or any(
                not isinstance(item, CalibrationCluster)
                for item in self.calibration_clusters
            )
        ):
            raise PrototypePairCohortError("calibration cluster count differs")
        by_tag = {
            tag_id: tuple(
                item for item in self.calibration_clusters if item.group_tag_id == tag_id
            )
            for tag_id in OPAQUE_TAG_IDS
        }
        if any(len(values) != CALIBRATION_CLUSTERS_PER_TAG for values in by_tag.values()):
            raise PrototypePairCohortError("per-tag calibration count differs")
        if any(
            tuple(item.selection_rank for item in values)
            != tuple(sorted(item.selection_rank for item in values))
            for values in by_tag.values()
        ):
            raise PrototypePairCohortError("calibration clusters are not rank ordered")
        selected_ids = (
            {item.task_id for item in self.prototypes}
            | {item.task_id for item in self.calibration_clusters}
            | {self.drill.task_id}
        )
        if len(selected_ids) != 2 + len(self.calibration_clusters) + 1:
            raise PrototypePairCohortError("prototype, calibration, and drill tasks overlap")
        target_a, target_b = self.drill.ordered_shapes
        for item in by_tag[OPAQUE_TAG_IDS[0]]:
            if target_a not in item.ordered_shapes or target_b in item.ordered_shapes:
                raise PrototypePairCohortError("tag-0 cross-calibration label is unclean")
        for item in by_tag[OPAQUE_TAG_IDS[1]]:
            if target_b not in item.ordered_shapes or target_a in item.ordered_shapes:
                raise PrototypePairCohortError("tag-1 cross-calibration label is unclean")
        if self.drill.task_id not in {item.task_id for item in self.candidates}:
            raise PrototypePairCohortError("selected drill is outside candidate inventory")
        expected_bound = familywise_clopper_pearson_upper_ppm(
            cluster_count=CALIBRATION_CLUSTERS_PER_TAG,
            error_cluster_count=0,
            confidence_level_ppm=CONFIDENCE_LEVEL_PPM,
            hypothesis_count=HYPOTHESIS_COUNT,
        )
        if expected_bound != ZERO_ERROR_FAMILY_UPPER_PPM:
            raise PrototypePairCohortError("frozen statistical convention drifted")
        if (
            self.hypothesis_count != HYPOTHESIS_COUNT
            or self.clusters_per_hypothesis != CALIBRATION_CLUSTERS_PER_TAG
            or self.confidence_level_ppm != CONFIDENCE_LEVEL_PPM
            or self.zero_error_family_upper_ppm != expected_bound
            or self.targeted_engineering_tolerance_ppm
            != TARGETED_ENGINEERING_TOLERANCE_PPM
            or self.zero_error_family_upper_ppm
            > self.targeted_engineering_tolerance_ppm
            or self.zero_errors_required_for_tolerance is not True
            or self.stronger_250k_claim_authorized is not False
            or self.thresholds_must_be_frozen_before_calibration is not True
        ):
            raise PrototypePairCohortError("statistical claim policy differs")
        if (
            self.upstream_repository != OFFICIAL_UPSTREAM_REPOSITORY
            or self.upstream_commit != OFFICIAL_UPSTREAM_COMMIT
            or self.basic_sampler_sha256 != OFFICIAL_BASIC_SAMPLER_SHA256
            or self.basic_generator_sha256 != OFFICIAL_BASIC_GENERATOR_SHA256
            or self.weak_label_authority != WEAK_LABEL_AUTHORITY
            or self.engineering_claim != ENGINEERING_CLAIM
            or self.drill_semantics_reused is not True
            or self.benchmark_claim_authorized is not False
            or self.unseen_claim_authorized is not False
            or self.validation_split_authorized is not False
            or self.official_test_authorized is not False
            or self.panel_bytes_read is not False
            or self.panel_paths_resolved is not False
            or self.action_program_json_authorized is not False
            or self.action_program_json_read is not False
            or self.planner_source_sha256 != PLANNER_SOURCE_SHA256
            or self.planner_algorithm_digest != planner_algorithm_digest()
            or self.predicate_authority_id != PYTHON_AUTHORITY_ID
            or self.python_is_canonical_authority is not True
            or self.lean_required is not False
            or self.lean_defines_artifact_identity is not False
            or self.lean_affects_selection_or_decision is not False
            or self.lean_required_for_replay is not False
            or self.optional_secondary_checker_detachable is not True
            or self.algorithm_id != ALGORITHM_ID
        ):
            raise PrototypePairCohortError("scientific or runtime authority differs")

    @property
    def selected_task_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                [item.task_id for item in self.prototypes]
                + [item.task_id for item in self.calibration_clusters]
                + [self.drill.task_id]
            )
        )

    def _hypothesis_clusters(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for tag_id in OPAQUE_TAG_IDS:
            for direction in ("present", "absent"):
                task_ids = [
                    item.task_id
                    for item in self.calibration_clusters
                    if dict(item.expected_tag_states)[tag_id] == direction
                ]
                rows.append(
                    {
                        "tag_id": tag_id,
                        "direction": direction,
                        "cluster_ids": task_ids,
                        "cluster_count": len(task_ids),
                    }
                )
        return rows

    def content_dict(self) -> dict[str, object]:
        return {
            "schema": PLAN_SCHEMA,
            "algorithm_id": self.algorithm_id,
            "namespace": self.namespace,
            "selection": {
                "selection_seed_digest": self.selection_seed_digest,
                "selection_seed_commitment": self.selection_seed_commitment,
                "candidate_count": len(self.candidates),
                "candidates": [item.to_data() for item in self.candidates],
                "candidate_task_ids_digest": _address(
                    [item.task_id for item in self.candidates]
                ),
                "bird_candidate_task_ids": list(self.bird_candidate_task_ids),
                "bird_candidate_task_ids_digest": _address(
                    list(self.bird_candidate_task_ids)
                ),
                "bird_candidates_reported_without_selection_preference": True,
                "excluded_exact_used_train_basic_task_ids": list(
                    self.excluded_exact_used_train_basic_task_ids
                ),
                "excluded_exact_used_digest": _address(
                    list(self.excluded_exact_used_train_basic_task_ids)
                ),
                "selected_task_ids": list(self.selected_task_ids),
                "selected_task_ids_digest": _address(list(self.selected_task_ids)),
            },
            "source_authority": {
                "release_descriptor_digest": self.release_descriptor_digest,
                "release_id": self.release_id,
                "archive_sha256": self.archive_sha256,
                "corpus_manifest_digest": self.corpus_manifest_digest,
                "split_source_digest": self.split_source_digest,
                "split_metadata_digest": self.split_metadata_digest,
                "task_inventory_digest": self.task_inventory_digest,
                "task_inventory_count": self.task_inventory_count,
                "historical_seed_digest": self.historical_seed_digest,
                "resolver_policy_digest": self.resolver_policy_digest,
                "exposure_predecessor_digest": self.exposure_predecessor_digest,
                "active_semantic_resolution_digest": (
                    self.active_semantic_resolution_digest
                ),
                "upstream_repository": self.upstream_repository,
                "upstream_commit": self.upstream_commit,
                "source_files": {
                    "bongard/sampler/basic_sampler.py": self.basic_sampler_sha256,
                    "examples/02-bongard_logo/generate_basic_problems.py": (
                        self.basic_generator_sha256
                    ),
                },
            },
            "prototype_bindings": [item.to_data() for item in self.prototypes],
            "calibration": {
                "clusters": [item.to_data() for item in self.calibration_clusters],
                "score_both_tags_on_every_panel": True,
                "hypotheses": self._hypothesis_clusters(),
                "hypothesis_count": self.hypothesis_count,
                "clusters_per_hypothesis": self.clusters_per_hypothesis,
                "confidence_level_ppm": self.confidence_level_ppm,
                "zero_error_family_upper_ppm": self.zero_error_family_upper_ppm,
                "targeted_engineering_tolerance_ppm": (
                    self.targeted_engineering_tolerance_ppm
                ),
                "zero_errors_required_for_tolerance": (
                    self.zero_errors_required_for_tolerance
                ),
                "stronger_250k_claim_authorized": (
                    self.stronger_250k_claim_authorized
                ),
                "thresholds_must_be_frozen_before_calibration": (
                    self.thresholds_must_be_frozen_before_calibration
                ),
            },
            "drill": self.drill.to_data(),
            "claim_scope": {
                "weak_label_authority": self.weak_label_authority,
                "engineering_claim": self.engineering_claim,
                "drill_semantics_reused": self.drill_semantics_reused,
                "benchmark_claim_authorized": self.benchmark_claim_authorized,
                "unseen_claim_authorized": self.unseen_claim_authorized,
                "validation_split_authorized": self.validation_split_authorized,
                "official_test_authorized": self.official_test_authorized,
            },
            "planning_input_boundary": {
                "accepted_release_data": "split-bytes-and-task-id-inventory-only",
                "panel_bytes_read": self.panel_bytes_read,
                "panel_paths_resolved": self.panel_paths_resolved,
                "action_program_json_authorized": self.action_program_json_authorized,
                "action_program_json_read": self.action_program_json_read,
            },
            "runtime_authority": {
                "planner_source_sha256": self.planner_source_sha256,
                "planner_algorithm_digest": self.planner_algorithm_digest,
                "predicate_authority_id": self.predicate_authority_id,
                "python_is_canonical_authority": self.python_is_canonical_authority,
                "lean_required": self.lean_required,
                "lean_defines_artifact_identity": self.lean_defines_artifact_identity,
                "lean_affects_selection_or_decision": (
                    self.lean_affects_selection_or_decision
                ),
                "lean_required_for_replay": self.lean_required_for_replay,
                "optional_secondary_checker_detachable": (
                    self.optional_secondary_checker_detachable
                ),
            },
        }

    @property
    def record_digest(self) -> str:
        return _address(self.content_dict())

    def to_data(self) -> dict[str, object]:
        return {**self.content_dict(), "record_digest": self.record_digest}

    @classmethod
    def from_data(cls, value: Mapping[str, Any]) -> "PrototypePairCohortPlan":
        """Reconstruct one archived plan without a live planner object."""

        root_fields = {
            "schema",
            "algorithm_id",
            "namespace",
            "selection",
            "source_authority",
            "prototype_bindings",
            "calibration",
            "drill",
            "claim_scope",
            "planning_input_boundary",
            "runtime_authority",
            "record_digest",
        }
        raw = _strict_object(value, root_fields, "prototype-pair plan")
        _verify_serialized_digest(raw, label="prototype-pair plan")
        if raw["schema"] != PLAN_SCHEMA:
            raise PrototypePairCohortError("prototype-pair plan schema differs")

        selection = _strict_object(
            raw["selection"],
            {
                "selection_seed_digest",
                "selection_seed_commitment",
                "candidate_count",
                "candidates",
                "candidate_task_ids_digest",
                "bird_candidate_task_ids",
                "bird_candidate_task_ids_digest",
                "bird_candidates_reported_without_selection_preference",
                "excluded_exact_used_train_basic_task_ids",
                "excluded_exact_used_digest",
                "selected_task_ids",
                "selected_task_ids_digest",
            },
            "plan selection",
        )
        source = _strict_object(
            raw["source_authority"],
            {
                "release_descriptor_digest",
                "release_id",
                "archive_sha256",
                "corpus_manifest_digest",
                "split_source_digest",
                "split_metadata_digest",
                "task_inventory_digest",
                "task_inventory_count",
                "historical_seed_digest",
                "resolver_policy_digest",
                "exposure_predecessor_digest",
                "active_semantic_resolution_digest",
                "upstream_repository",
                "upstream_commit",
                "source_files",
            },
            "plan source authority",
        )
        source_files = _strict_object(
            source["source_files"],
            {
                "bongard/sampler/basic_sampler.py",
                "examples/02-bongard_logo/generate_basic_problems.py",
            },
            "plan source files",
        )
        calibration = _strict_object(
            raw["calibration"],
            {
                "clusters",
                "score_both_tags_on_every_panel",
                "hypotheses",
                "hypothesis_count",
                "clusters_per_hypothesis",
                "confidence_level_ppm",
                "zero_error_family_upper_ppm",
                "targeted_engineering_tolerance_ppm",
                "zero_errors_required_for_tolerance",
                "stronger_250k_claim_authorized",
                "thresholds_must_be_frozen_before_calibration",
            },
            "plan calibration",
        )
        claim = _strict_object(
            raw["claim_scope"],
            {
                "weak_label_authority",
                "engineering_claim",
                "drill_semantics_reused",
                "benchmark_claim_authorized",
                "unseen_claim_authorized",
                "validation_split_authorized",
                "official_test_authorized",
            },
            "plan claim scope",
        )
        boundary = _strict_object(
            raw["planning_input_boundary"],
            {
                "accepted_release_data",
                "panel_bytes_read",
                "panel_paths_resolved",
                "action_program_json_authorized",
                "action_program_json_read",
            },
            "plan input boundary",
        )
        runtime = _strict_object(
            raw["runtime_authority"],
            {
                "predicate_authority_id",
                "planner_source_sha256",
                "planner_algorithm_digest",
                "python_is_canonical_authority",
                "lean_required",
                "lean_defines_artifact_identity",
                "lean_affects_selection_or_decision",
                "lean_required_for_replay",
                "optional_secondary_checker_detachable",
            },
            "plan runtime authority",
        )
        candidates = tuple(
            CandidateSummary.from_data(item)
            for item in _strict_list(selection["candidates"], "plan candidates")
        )
        prototypes = tuple(
            PrototypeBinding.from_data(item)
            for item in _strict_list(
                raw["prototype_bindings"], "plan prototype bindings"
            )
        )
        clusters = tuple(
            CalibrationCluster.from_data(item)
            for item in _strict_list(calibration["clusters"], "plan clusters")
        )
        result = cls(
            namespace=raw["namespace"],
            selection_seed_digest=selection["selection_seed_digest"],
            selection_seed_commitment=selection["selection_seed_commitment"],
            release_descriptor_digest=source["release_descriptor_digest"],
            release_id=source["release_id"],
            archive_sha256=source["archive_sha256"],
            corpus_manifest_digest=source["corpus_manifest_digest"],
            split_source_digest=source["split_source_digest"],
            split_metadata_digest=source["split_metadata_digest"],
            task_inventory_digest=source["task_inventory_digest"],
            task_inventory_count=source["task_inventory_count"],
            historical_seed_digest=source["historical_seed_digest"],
            resolver_policy_digest=source["resolver_policy_digest"],
            exposure_predecessor_digest=source[
                "exposure_predecessor_digest"
            ],
            active_semantic_resolution_digest=source[
                "active_semantic_resolution_digest"
            ],
            upstream_repository=source["upstream_repository"],
            upstream_commit=source["upstream_commit"],
            basic_sampler_sha256=source_files[
                "bongard/sampler/basic_sampler.py"
            ],
            basic_generator_sha256=source_files[
                "examples/02-bongard_logo/generate_basic_problems.py"
            ],
            candidates=candidates,
            bird_candidate_task_ids=tuple(
                _strict_list(
                    selection["bird_candidate_task_ids"],
                    "bird candidate task IDs",
                )
            ),
            excluded_exact_used_train_basic_task_ids=tuple(
                _strict_list(
                    selection["excluded_exact_used_train_basic_task_ids"],
                    "excluded exact-used tasks",
                )
            ),
            prototypes=prototypes,  # type: ignore[arg-type]
            calibration_clusters=clusters,
            drill=DrillSchedule.from_data(raw["drill"]),
            hypothesis_count=calibration["hypothesis_count"],
            clusters_per_hypothesis=calibration[
                "clusters_per_hypothesis"
            ],
            confidence_level_ppm=calibration["confidence_level_ppm"],
            zero_error_family_upper_ppm=calibration[
                "zero_error_family_upper_ppm"
            ],
            targeted_engineering_tolerance_ppm=calibration[
                "targeted_engineering_tolerance_ppm"
            ],
            zero_errors_required_for_tolerance=calibration[
                "zero_errors_required_for_tolerance"
            ],
            stronger_250k_claim_authorized=calibration[
                "stronger_250k_claim_authorized"
            ],
            thresholds_must_be_frozen_before_calibration=calibration[
                "thresholds_must_be_frozen_before_calibration"
            ],
            weak_label_authority=claim["weak_label_authority"],
            engineering_claim=claim["engineering_claim"],
            drill_semantics_reused=claim["drill_semantics_reused"],
            benchmark_claim_authorized=claim["benchmark_claim_authorized"],
            unseen_claim_authorized=claim["unseen_claim_authorized"],
            validation_split_authorized=claim[
                "validation_split_authorized"
            ],
            official_test_authorized=claim["official_test_authorized"],
            panel_bytes_read=boundary["panel_bytes_read"],
            panel_paths_resolved=boundary["panel_paths_resolved"],
            action_program_json_authorized=boundary[
                "action_program_json_authorized"
            ],
            action_program_json_read=boundary["action_program_json_read"],
            planner_source_sha256=runtime["planner_source_sha256"],
            planner_algorithm_digest=runtime["planner_algorithm_digest"],
            predicate_authority_id=runtime["predicate_authority_id"],
            python_is_canonical_authority=runtime[
                "python_is_canonical_authority"
            ],
            lean_required=runtime["lean_required"],
            lean_defines_artifact_identity=runtime[
                "lean_defines_artifact_identity"
            ],
            lean_affects_selection_or_decision=runtime[
                "lean_affects_selection_or_decision"
            ],
            lean_required_for_replay=runtime["lean_required_for_replay"],
            optional_secondary_checker_detachable=runtime[
                "optional_secondary_checker_detachable"
            ],
            algorithm_id=raw["algorithm_id"],
        )
        if result.to_data() != dict(raw):
            raise PrototypePairCohortError(
                "prototype-pair plan is not the strict canonical form"
            )
        return result


def _source_checks(
    *,
    release_descriptor: OfficialReleaseDescriptor,
    split_bytes: bytes,
    task_ids: tuple[str, ...],
    exposure_predecessor: ExposureLedger,
    historical_seed: HistoricalExposureSeed,
    expected_release_descriptor_digest: str,
    expected_corpus_manifest_digest: str,
    expected_split_source_digest: str,
    expected_task_inventory_digest: str,
    expected_exposure_predecessor_digest: str,
    expected_historical_seed_digest: str,
    expected_resolver_policy_digest: str,
    expected_basic_sampler_sha256: str,
    expected_basic_generator_sha256: str,
) -> tuple[SplitIndex, str, str]:
    if not isinstance(release_descriptor, OfficialReleaseDescriptor):
        raise TypeError("release_descriptor must be OfficialReleaseDescriptor")
    if release_descriptor.digest != _require_address(
        expected_release_descriptor_digest, "release descriptor digest"
    ):
        raise PrototypePairCohortError("release descriptor differs from external pin")
    corpus_pin = _require_address(
        expected_corpus_manifest_digest, "corpus manifest digest"
    )
    if release_descriptor.corpus_manifest_sha256 != corpus_pin:
        raise PrototypePairCohortError("release corpus identity differs from pin")
    split_pin = _require_address(expected_split_source_digest, "split source digest")
    if release_descriptor.split_sha256 != split_pin:
        raise PrototypePairCohortError("release split identity differs from pin")
    inventory_pin = _require_address(
        expected_task_inventory_digest, "task inventory digest"
    )
    if (
        release_descriptor.task_ids_sha256 != inventory_pin
        or task_id_inventory_digest(task_ids) != inventory_pin
    ):
        raise PrototypePairCohortError("task-ID inventory differs from release authority")
    if (
        release_descriptor.upstream_repository != OFFICIAL_UPSTREAM_REPOSITORY
        or release_descriptor.upstream_commit != OFFICIAL_UPSTREAM_COMMIT
    ):
        raise PrototypePairCohortError("release upstream source identity differs")
    if (
        _require_raw_sha256(
            expected_basic_sampler_sha256, "expected Basic sampler SHA-256"
        )
        != OFFICIAL_BASIC_SAMPLER_SHA256
        or _require_raw_sha256(
            expected_basic_generator_sha256, "expected Basic generator SHA-256"
        )
        != OFFICIAL_BASIC_GENERATOR_SHA256
    ):
        raise PrototypePairCohortError("Basic source pin differs")
    split = _authenticated_split(
        split_bytes,
        expected_digest=split_pin,
        expected_size=release_descriptor.split_size_bytes,
    )
    try:
        split.validate(task_ids)
    except (CorpusError, TypeError, ValueError) as exc:
        raise PrototypePairCohortError(f"split/task inventory differs: {exc}") from exc
    groups = split.canonical_groups
    primary_counts = {name: len(groups[name]) for name in ("train", "val", "test")}
    regime_counts = {name: len(groups[name]) for name in ("BA", "CM", "FF", "NV")}
    if primary_counts != dict(release_descriptor.primary_split_counts):
        raise PrototypePairCohortError("primary split counts differ from release")
    if regime_counts != dict(release_descriptor.regime_counts):
        raise PrototypePairCohortError("test regime counts differ from release")
    family_counts = Counter(
        task_id[:2]
        for task_id in task_ids
        if len(task_id) >= 3 and task_id[2] == "_" and task_id[:2] in {"bd", "hd", "ff"}
    )
    if sum(family_counts.values()) != len(task_ids) or dict(family_counts) != dict(
        release_descriptor.family_counts
    ):
        raise PrototypePairCohortError("task family counts differ from release")
    if not isinstance(exposure_predecessor, ExposureLedger):
        raise TypeError("exposure_predecessor must be ExposureLedger")
    if exposure_predecessor.digest != _require_address(
        expected_exposure_predecessor_digest, "exposure predecessor digest"
    ):
        raise PrototypePairCohortError("exposure predecessor differs from pin")
    exposure_predecessor.assert_corpus(corpus_pin)
    if not isinstance(historical_seed, HistoricalExposureSeed):
        raise TypeError("historical_seed must be HistoricalExposureSeed")
    if historical_seed.seed_digest != _require_address(
        expected_historical_seed_digest, "historical seed digest"
    ):
        raise PrototypePairCohortError("historical seed differs from pin")
    resolver = semantic_resolver_policy_digest(historical_seed)
    if resolver != _require_address(
        expected_resolver_policy_digest, "semantic resolver policy digest"
    ):
        raise PrototypePairCohortError("semantic resolver differs from pin")
    active = exposure_predecessor.derive_exposed_semantic_keys(
        historical_seed=historical_seed,
        expected_historical_seed_digest=historical_seed.seed_digest,
        expected_resolver_policy_digest=resolver,
    )
    return split, _address(split.to_manifest_dict()), _address(active.to_dict())


def _exact_used_task_ids(
    historical_seed: HistoricalExposureSeed,
    exposure_predecessor: ExposureLedger,
) -> frozenset[str]:
    historical_panel_tasks = {
        task_id_from_panel_id(panel_id)
        for panel_id in historical_seed.exact_official_panel_ids
    }
    return frozenset(
        set(historical_seed.exact_official_task_ids)
        | historical_panel_tasks
        | set(exposure_predecessor.exposed_task_ids)
    )


def plan_prototype_pair_cohort(
    *,
    release_descriptor: OfficialReleaseDescriptor,
    split_bytes: bytes,
    task_ids: Sequence[str],
    exposure_predecessor: ExposureLedger,
    historical_seed: HistoricalExposureSeed,
    selection_seed: str,
    expected_seed_commitment: str,
    expected_release_descriptor_digest: str,
    expected_corpus_manifest_digest: str,
    expected_split_source_digest: str,
    expected_task_inventory_digest: str,
    expected_exposure_predecessor_digest: str,
    expected_historical_seed_digest: str,
    expected_resolver_policy_digest: str,
    expected_basic_sampler_sha256: str,
    expected_basic_generator_sha256: str,
    namespace: str = DEFAULT_NAMESPACE,
) -> PrototypePairCohortPlan:
    """Freeze one drill and its prototype/cross-calibration schedules."""

    seed = _selection_seed(selection_seed)
    _require_identifier(namespace, "namespace")
    commitment = _require_address(expected_seed_commitment, "seed commitment")
    if commitment != prototype_pair_seed_commitment(seed, namespace=namespace):
        raise PrototypePairCohortError("selection seed differs from external commitment")
    inventory = tuple(task_ids)
    split, split_metadata_digest, active_resolution_digest = _source_checks(
        release_descriptor=release_descriptor,
        split_bytes=split_bytes,
        task_ids=inventory,
        exposure_predecessor=exposure_predecessor,
        historical_seed=historical_seed,
        expected_release_descriptor_digest=expected_release_descriptor_digest,
        expected_corpus_manifest_digest=expected_corpus_manifest_digest,
        expected_split_source_digest=expected_split_source_digest,
        expected_task_inventory_digest=expected_task_inventory_digest,
        expected_exposure_predecessor_digest=expected_exposure_predecessor_digest,
        expected_historical_seed_digest=expected_historical_seed_digest,
        expected_resolver_policy_digest=expected_resolver_policy_digest,
        expected_basic_sampler_sha256=expected_basic_sampler_sha256,
        expected_basic_generator_sha256=expected_basic_generator_sha256,
    )
    vocabulary = frozenset(historical_seed.basic_shape_families) | frozenset(
        historical_seed.unused_basic_shape_families
    )
    parsed_basic: dict[str, tuple[str, ...]] = {
        task_id: _parse_basic_task_id(task_id, vocabulary)
        for task_id in inventory
        if task_id.startswith("bd_")
    }
    train_basic = {
        task_id: shapes
        for task_id, shapes in parsed_basic.items()
        if split.assignment(task_id).split == "train"
        and split.assignment(task_id).regime is None
    }
    exact_used = _exact_used_task_ids(historical_seed, exposure_predecessor)
    excluded = tuple(sorted(set(train_basic) & set(exact_used)))
    exact_unused = {
        task_id: shapes
        for task_id, shapes in train_basic.items()
        if task_id not in exact_used
    }
    unused_singletons = {
        shapes[0]: task_id
        for task_id, shapes in exact_unused.items()
        if len(shapes) == 1
    }
    unused_pairs = {
        task_id: (shapes[0], shapes[1])
        for task_id, shapes in exact_unused.items()
        if len(shapes) == 2
    }
    pairs_by_shape: dict[str, list[str]] = defaultdict(list)
    pairs_by_unordered_shapes: dict[frozenset[str], list[str]] = defaultdict(list)
    for task_id, shapes in unused_pairs.items():
        pairs_by_shape[shapes[0]].append(task_id)
        pairs_by_shape[shapes[1]].append(task_id)
        pairs_by_unordered_shapes[frozenset(shapes)].append(task_id)

    candidate_rows: list[CandidateSummary] = []
    eligible_calibration_by_candidate: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {}
    for task_id, (shape_a, shape_b) in sorted(unused_pairs.items()):
        if shape_a not in unused_singletons or shape_b not in unused_singletons:
            continue
        if pairs_by_unordered_shapes[frozenset((shape_a, shape_b))] != [task_id]:
            continue
        other_a = tuple(
            sorted(
                other_task_id
                for other_task_id in pairs_by_shape[shape_a]
                if other_task_id != task_id and shape_b not in unused_pairs[other_task_id]
            )
        )
        other_b = tuple(
            sorted(
                other_task_id
                for other_task_id in pairs_by_shape[shape_b]
                if other_task_id != task_id and shape_a not in unused_pairs[other_task_id]
            )
        )
        if (
            len(other_a) < MIN_CALIBRATION_CLUSTERS
            or len(other_b) < MIN_CALIBRATION_CLUSTERS
        ):
            continue
        candidate_rows.append(
            CandidateSummary(
                task_id=task_id,
                ordered_shapes=(shape_a, shape_b),
                first_shape_other_cluster_count=len(other_a),
                second_shape_other_cluster_count=len(other_b),
                unique_joint_task=True,
                both_prototypes_exact_unused=True,
                bird_family_matches=tuple(
                    shape for shape in (shape_a, shape_b) if shape in BIRD_FAMILIES
                ),
            )
        )
        eligible_calibration_by_candidate[task_id] = (other_a, other_b)
    candidates = tuple(candidate_rows)
    if not candidates:
        raise PrototypePairCohortError(
            "no exact-unused TRAIN Basic pair has two exact-unused singleton "
            "prototypes and at least 14 clean other pair clusters per shape"
        )

    seed_digest = _seed_digest(seed)
    selected_candidate = min(
        candidates,
        key=lambda item: (
            _rank(
                namespace=namespace,
                seed_digest=seed_digest,
                role="drill-candidate",
                task_id=item.task_id,
            ),
            item.task_id,
        ),
    )
    drill_rank = _rank(
        namespace=namespace,
        seed_digest=seed_digest,
        role="drill-candidate",
        task_id=selected_candidate.task_id,
    )
    shape_a, shape_b = selected_candidate.ordered_shapes
    prototypes = tuple(
        PrototypeBinding(
            tag_id=tag_id,
            shape_family=shape,
            task_id=unused_singletons[shape],
            cluster_id=unused_singletons[shape],
            side="positive",
            panel_indices=PROTOTYPE_POSITIVE_INDICES,
            panel_ids=tuple(
                _panel_id(unused_singletons[shape], "positive", index)
                for index in PROTOTYPE_POSITIVE_INDICES
            ),
            exact_task_unused=True,
            weak_generator_identity_label=True,
        )
        for tag_id, shape in zip(OPAQUE_TAG_IDS, (shape_a, shape_b), strict=True)
    )
    calibration: list[CalibrationCluster] = []
    available_groups = eligible_calibration_by_candidate[selected_candidate.task_id]
    for group_index, (tag_id, task_group) in enumerate(
        zip(OPAQUE_TAG_IDS, available_groups, strict=True)
    ):
        ranked = sorted(
            (
                _rank(
                    namespace=namespace,
                    seed_digest=seed_digest,
                    role="calibration-cluster",
                    tag_id=tag_id,
                    task_id=task_id,
                ),
                task_id,
            )
            for task_id in task_group
        )[:CALIBRATION_CLUSTERS_PER_TAG]
        states = tuple(
            (candidate_tag, "present" if tag_position == group_index else "absent")
            for tag_position, candidate_tag in enumerate(OPAQUE_TAG_IDS)
        )
        for selection_rank, task_id in ranked:
            index = _calibration_index(
                namespace=namespace,
                seed_digest=seed_digest,
                tag_id=tag_id,
                task_id=task_id,
            )
            calibration.append(
                CalibrationCluster(
                    task_id=task_id,
                    ordered_shapes=unused_pairs[task_id],
                    cluster_id=task_id,
                    side="positive",
                    panel_index=index,
                    panel_id=_panel_id(task_id, "positive", index),
                    score_tag_ids=OPAQUE_TAG_IDS,
                    expected_tag_states=states,  # type: ignore[arg-type]
                    group_tag_id=tag_id,
                    selection_rank=selection_rank,
                    exact_task_unused=True,
                )
            )
    drill = DrillSchedule(
        task_id=selected_candidate.task_id,
        ordered_shapes=selected_candidate.ordered_shapes,
        cluster_id=selected_candidate.task_id,
        selection_rank=drill_rank,
        positive_indices=DRILL_POSITIVE_INDICES,
        negative_indices=DRILL_NEGATIVE_INDICES,
        positive_panel_ids=tuple(
            _panel_id(selected_candidate.task_id, "positive", index)
            for index in DRILL_POSITIVE_INDICES
        ),
        negative_panel_ids=tuple(
            _panel_id(selected_candidate.task_id, "negative", index)
            for index in DRILL_NEGATIVE_INDICES
        ),
        exact_task_unused=True,
        pixels_opened_during_planning=False,
    )
    plan = PrototypePairCohortPlan(
        namespace=namespace,
        selection_seed_digest=seed_digest,
        selection_seed_commitment=commitment,
        release_descriptor_digest=release_descriptor.digest,
        release_id=release_descriptor.release_id,
        archive_sha256=release_descriptor.archive_sha256,
        corpus_manifest_digest=release_descriptor.corpus_manifest_sha256,
        split_source_digest=release_descriptor.split_sha256,
        split_metadata_digest=split_metadata_digest,
        task_inventory_digest=release_descriptor.task_ids_sha256,
        task_inventory_count=len(inventory),
        historical_seed_digest=historical_seed.seed_digest,
        resolver_policy_digest=semantic_resolver_policy_digest(historical_seed),
        exposure_predecessor_digest=exposure_predecessor.digest,
        active_semantic_resolution_digest=active_resolution_digest,
        upstream_repository=release_descriptor.upstream_repository,
        upstream_commit=release_descriptor.upstream_commit,
        basic_sampler_sha256=OFFICIAL_BASIC_SAMPLER_SHA256,
        basic_generator_sha256=OFFICIAL_BASIC_GENERATOR_SHA256,
        candidates=candidates,
        bird_candidate_task_ids=tuple(
            item.task_id for item in candidates if item.bird_family_matches
        ),
        excluded_exact_used_train_basic_task_ids=excluded,
        prototypes=prototypes,  # type: ignore[arg-type]
        calibration_clusters=tuple(calibration),
        drill=drill,
        hypothesis_count=HYPOTHESIS_COUNT,
        clusters_per_hypothesis=CALIBRATION_CLUSTERS_PER_TAG,
        confidence_level_ppm=CONFIDENCE_LEVEL_PPM,
        zero_error_family_upper_ppm=ZERO_ERROR_FAMILY_UPPER_PPM,
        targeted_engineering_tolerance_ppm=TARGETED_ENGINEERING_TOLERANCE_PPM,
        zero_errors_required_for_tolerance=True,
        stronger_250k_claim_authorized=False,
        thresholds_must_be_frozen_before_calibration=True,
        weak_label_authority=WEAK_LABEL_AUTHORITY,
        engineering_claim=ENGINEERING_CLAIM,
        drill_semantics_reused=True,
        benchmark_claim_authorized=False,
        unseen_claim_authorized=False,
        validation_split_authorized=False,
        official_test_authorized=False,
        panel_bytes_read=False,
        panel_paths_resolved=False,
        action_program_json_authorized=False,
        action_program_json_read=False,
        planner_source_sha256=PLANNER_SOURCE_SHA256,
        planner_algorithm_digest=planner_algorithm_digest(),
        predicate_authority_id=PYTHON_AUTHORITY_ID,
        python_is_canonical_authority=True,
        lean_required=False,
        lean_defines_artifact_identity=False,
        lean_affects_selection_or_decision=False,
        lean_required_for_replay=False,
        optional_secondary_checker_detachable=True,
        algorithm_id=ALGORITHM_ID,
    )
    if any(task_id in exact_used for task_id in plan.selected_task_ids):
        raise PrototypePairCohortError("selected schedule contains an exact-used task")
    return plan


def verify_prototype_pair_cohort_plan(
    plan: PrototypePairCohortPlan | Mapping[str, Any],
    *,
    expected_plan_digest: str,
    release_descriptor: OfficialReleaseDescriptor,
    split_bytes: bytes,
    task_ids: Sequence[str],
    exposure_predecessor: ExposureLedger,
    historical_seed: HistoricalExposureSeed,
    selection_seed: str,
    expected_seed_commitment: str,
    expected_release_descriptor_digest: str,
    expected_corpus_manifest_digest: str,
    expected_split_source_digest: str,
    expected_task_inventory_digest: str,
    expected_exposure_predecessor_digest: str,
    expected_historical_seed_digest: str,
    expected_resolver_policy_digest: str,
    expected_basic_sampler_sha256: str,
    expected_basic_generator_sha256: str,
) -> PrototypePairCohortPlan:
    """Cold-recompute the complete plan from the authenticated metadata."""

    archived = (
        plan
        if isinstance(plan, PrototypePairCohortPlan)
        else PrototypePairCohortPlan.from_data(plan)
    )
    if archived.record_digest != _require_address(expected_plan_digest, "plan digest"):
        raise PrototypePairCohortError("plan differs from external commitment")
    replay = plan_prototype_pair_cohort(
        release_descriptor=release_descriptor,
        split_bytes=split_bytes,
        task_ids=task_ids,
        exposure_predecessor=exposure_predecessor,
        historical_seed=historical_seed,
        selection_seed=selection_seed,
        expected_seed_commitment=expected_seed_commitment,
        expected_release_descriptor_digest=expected_release_descriptor_digest,
        expected_corpus_manifest_digest=expected_corpus_manifest_digest,
        expected_split_source_digest=expected_split_source_digest,
        expected_task_inventory_digest=expected_task_inventory_digest,
        expected_exposure_predecessor_digest=expected_exposure_predecessor_digest,
        expected_historical_seed_digest=expected_historical_seed_digest,
        expected_resolver_policy_digest=expected_resolver_policy_digest,
        expected_basic_sampler_sha256=expected_basic_sampler_sha256,
        expected_basic_generator_sha256=expected_basic_generator_sha256,
        namespace=archived.namespace,
    )
    if replay != archived or replay.record_digest != archived.record_digest:
        raise PrototypePairCohortError("cold-recomputed cohort plan differs")
    return archived


__all__ = [
    "ALGORITHM_ID",
    "BIRD_FAMILIES",
    "CALIBRATION_CLUSTERS_PER_TAG",
    "CONFIDENCE_LEVEL_PPM",
    "DEFAULT_NAMESPACE",
    "ENGINEERING_CLAIM",
    "HYPOTHESIS_COUNT",
    "MIN_CALIBRATION_CLUSTERS",
    "OFFICIAL_BASIC_GENERATOR_SHA256",
    "OFFICIAL_BASIC_SAMPLER_SHA256",
    "OPAQUE_TAG_IDS",
    "PROTOTYPE_POSITIVE_INDICES",
    "PYTHON_AUTHORITY_ID",
    "PrototypePairCohortError",
    "PrototypePairCohortPlan",
    "TARGETED_ENGINEERING_TOLERANCE_PPM",
    "WEAK_LABEL_AUTHORITY",
    "ZERO_ERROR_FAMILY_UPPER_PPM",
    "plan_prototype_pair_cohort",
    "prototype_pair_seed_commitment",
    "task_id_inventory_digest",
    "verify_prototype_pair_cohort_plan",
]

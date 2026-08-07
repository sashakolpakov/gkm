"""Candidate-independent cross-scenario object lineages.

The lineage extractor sees exact panel bytes and the exhaustive low-level
object-hypothesis catalog only.  It does not see a Bongard side, label,
profile, rubric, or predicate.  A lineage is one reciprocal-unique mask match
in each of the three frozen segmentation scenarios.

Multicomponent candidates are not automatically objects.  They may contribute
downstream evidence only when the single-linkage cluster persists for a
strictly positive gap before joining an exterior component.  In particular,
the final whole-scene union has no exterior/death gap and remains unresolved.
"""

from __future__ import annotations

from bongard.runtime_source_snapshot import capture_loaded_source, verify_loaded_source


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)

from dataclasses import dataclass
from enum import Enum
import itertools
import re
from typing import Any, Mapping, Sequence

import numpy as np

from bongard.canonical import canonical_digest
from bongard.evidence import Disposition
from bongard.prototype_object_hypotheses import (
    ObjectHypothesis,
    ObjectHypothesisPacket,
    extract_object_hypothesis_packet,
    object_hypothesis_extractor_artifact_digest,
    verify_object_hypothesis_packet,
)
from bongard import prototype_object_hypotheses as _hypotheses
from bongard.prototype_object_profiles import (
    OBJECT_FEATURE_CATALOG,
    OBJECT_FEATURE_CATALOG_DIGEST,
    OBJECT_FEATURE_IDS,
    IntegerInterval,
    ObjectFeatureCell,
    ObjectFeatureCellState,
    ObjectLocalObservationPacket,
)
from bongard.python_predicate_authority import PYTHON_PREDICATE_AUTHORITY_ID
from bongard import visual_witnesses as _visual
from bongard.visual_witnesses import VISUAL_WITNESS_SCENARIO_IDS


OBJECT_LINEAGE_PACKET_SCHEMA = "gkm.bongard-object-lineage-packet.v1"
OBJECT_LINEAGE_AGGREGATION_SCHEMA = (
    "gkm.bongard-object-lineage-observation-aggregation.v1"
)
OBJECT_LINEAGE_FEATURE_SCHEMA = "gkm.bongard-object-lineage-feature-evidence.v1"
OBJECT_LINEAGE_OBSERVATION_SCHEMA = "gkm.bongard-object-lineage-observation.v1"
OBJECT_LINEAGE_ALGORITHM_ID = (
    "bongard.prototype-object-lineages/reciprocal-mask-iou-persistence-v1"
)
OBJECT_LINEAGE_AGGREGATION_ID = (
    "bongard.prototype-object-lineage-aggregation/same-lineage-envelope-v1"
)
MIN_MASK_IOU_PPM = 500_000
PPM_SCALE = 1_000_000

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_HYPOTHESIS_ID = re.compile(r"hypothesis-[0-9]{8}\Z")
_LINEAGE_ID = re.compile(r"lineage-[0-9]{8}\Z")


class ObjectLineageError(ValueError):
    """A lineage, observation aggregation, or replay binding is invalid."""


class LineageOwnershipState(str, Enum):
    SAFE_SINGLETON = "safe_singleton"
    SAFE_PERSISTENT_UNION = "safe_persistent_union"
    UNRESOLVED_UNION = "unresolved_union"


def _authority_data() -> dict[str, object]:
    return {
        "predicate_authority_id": PYTHON_PREDICATE_AUTHORITY_ID,
        "python_is_canonical_authority": True,
        "lean_present": False,
        "lean_required": False,
        "lean_removable": True,
        "lean_defines_identity_or_decision": False,
        "lean_required_for_replay": False,
        "lean_removal_changes_decision": False,
    }


def _linkage_policy_data() -> dict[str, object]:
    return {
        "scenario_ids": list(VISUAL_WITNESS_SCENARIO_IDS),
        "mask_similarity": "exact-source-pixel-intersection-over-union",
        "minimum_mask_iou_ppm": MIN_MASK_IOU_PPM,
        "bbox_requirement": "positive-half-open-pixel-intersection",
        "assignment": "reciprocal-unique-best-in-every-scenario-pair",
        "lineage_cardinality": "exactly-one-hypothesis-per-scenario",
        "singleton_eligible": True,
        "multicomponent_eligibility": (
            "emergence_gap_strictly_below_nearest-strict-superset-death-gap"
        ),
        "whole_scene_multicomponent_without_exterior": "unresolved_ineligible",
        "candidate_independent": True,
        "semantic_object_claimed": False,
        "labels_or_profiles_consumed": False,
    }


def _aggregation_policy_data() -> dict[str, object]:
    return {
        "same_lineage_required_across_all_scenarios": True,
        "lineage_interval": "min-lower-max-upper-envelope",
        "lineage_failure_precedence": ["error", "indeterminate", "scored"],
        "scene_reduction": "none; downstream existential keeps lineage identity",
        "ownership_unresolved_lineages_excluded": True,
        "empty_eligible_set": "indeterminate",
        "labels_or_profiles_consumed": False,
    }


def _exact_fields(value: object, expected: set[str], label: str) -> Mapping[str, Any]:
    if (
        not isinstance(value, Mapping)
        or any(not isinstance(key, str) for key in value)
        or set(value) != expected
    ):
        raise ObjectLineageError(f"{label} fields differ from schema")
    return value


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ObjectLineageError(f"{label} must be an integer >= {minimum}")
    return value


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ObjectLineageError(f"{label} must be a lowercase SHA-256")
    return value


def _bbox(value: object, label: str) -> tuple[int, int, int, int]:
    if (
        not isinstance(value, (tuple, list))
        or len(value) != 4
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
    ):
        raise ObjectLineageError(f"{label} must be a four-integer bbox")
    result = tuple(value)
    x0, y0, x1, y1 = result
    if min(x0, y0) < 0 or x1 <= x0 or y1 <= y0:
        raise ObjectLineageError(f"{label} must have positive half-open extent")
    return result  # type: ignore[return-value]


def object_lineage_source_digest() -> str:
    return verify_loaded_source(__name__, expected_source_sha256=_LOADED_SOURCE_SHA256)


def object_lineage_artifact_digest() -> str:
    return canonical_digest(
        {
            "algorithm_id": OBJECT_LINEAGE_ALGORITHM_ID,
            "source_digest": object_lineage_source_digest(),
            "object_hypothesis_extractor_artifact_digest": (
                object_hypothesis_extractor_artifact_digest()
            ),
            "visual_witness_extractor_artifact_digest": (
                _visual.visual_witness_extractor_digest()
            ),
            "linkage_policy": _linkage_policy_data(),
            "aggregation_policy": _aggregation_policy_data(),
            "feature_catalog_digest": OBJECT_FEATURE_CATALOG_DIGEST,
            "runtime_authority": _authority_data(),
        }
    )


@dataclass(frozen=True, order=True, slots=True)
class ObjectLineageMember:
    scenario_id: str
    hypothesis_id: str
    hypothesis_digest: str
    union_mask_digest: str
    union_area_pixels: int
    bbox_pixels: tuple[int, int, int, int]
    source_component_ids: tuple[str, ...]
    scenario_component_count: int
    emergence_gap_pixels: int
    nearest_external_gap_pixels: int | None

    def __post_init__(self) -> None:
        if self.scenario_id not in VISUAL_WITNESS_SCENARIO_IDS:
            raise ObjectLineageError("member scenario is not frozen")
        if not isinstance(self.hypothesis_id, str) or _HYPOTHESIS_ID.fullmatch(
            self.hypothesis_id
        ) is None:
            raise ObjectLineageError("member hypothesis ID is not canonical")
        _digest(self.hypothesis_digest, "member hypothesis digest")
        _digest(self.union_mask_digest, "member union mask digest")
        _integer(self.union_area_pixels, "member union area", minimum=1)
        _bbox(self.bbox_pixels, "member bbox")
        if (
            not isinstance(self.source_component_ids, tuple)
            or not self.source_component_ids
            or self.source_component_ids != tuple(sorted(set(self.source_component_ids)))
        ):
            raise ObjectLineageError("member component IDs are not canonical")
        count = _integer(self.scenario_component_count, "scenario component count", minimum=1)
        if len(self.source_component_ids) > count:
            raise ObjectLineageError("member owns more components than its scenario")
        emergence = _integer(self.emergence_gap_pixels, "member emergence gap")
        if (len(self.source_component_ids) == 1) != (emergence == 0):
            raise ObjectLineageError("only singleton members emerge at zero")
        if self.nearest_external_gap_pixels is not None:
            death = _integer(
                self.nearest_external_gap_pixels, "member nearest external gap"
            )
            if death < emergence:
                raise ObjectLineageError("member death gap precedes emergence")

    @property
    def is_singleton(self) -> bool:
        return len(self.source_component_ids) == 1

    @property
    def is_whole_scene_union(self) -> bool:
        return (
            not self.is_singleton
            and len(self.source_component_ids) == self.scenario_component_count
        )

    @property
    def persistent_before_external_merge(self) -> bool:
        return self.is_singleton or (
            self.nearest_external_gap_pixels is not None
            and self.emergence_gap_pixels < self.nearest_external_gap_pixels
        )

    def to_data(self) -> dict[str, object]:
        return {
            "scenario_id": self.scenario_id,
            "hypothesis_id": self.hypothesis_id,
            "hypothesis_digest": self.hypothesis_digest,
            "union_mask_digest": self.union_mask_digest,
            "union_area_pixels": self.union_area_pixels,
            "bbox_pixels": list(self.bbox_pixels),
            "source_component_ids": list(self.source_component_ids),
            "scenario_component_count": self.scenario_component_count,
            "emergence_gap_pixels": self.emergence_gap_pixels,
            "nearest_external_gap_pixels": self.nearest_external_gap_pixels,
            "is_singleton": self.is_singleton,
            "is_whole_scene_union": self.is_whole_scene_union,
            "persistent_before_external_merge": (
                self.persistent_before_external_merge
            ),
        }

    @classmethod
    def from_data(cls, value: object) -> "ObjectLineageMember":
        raw = _exact_fields(
            value,
            {
                "scenario_id",
                "hypothesis_id",
                "hypothesis_digest",
                "union_mask_digest",
                "union_area_pixels",
                "bbox_pixels",
                "source_component_ids",
                "scenario_component_count",
                "emergence_gap_pixels",
                "nearest_external_gap_pixels",
                "is_singleton",
                "is_whole_scene_union",
                "persistent_before_external_merge",
            },
            "lineage member",
        )
        if not isinstance(raw["source_component_ids"], list):
            raise ObjectLineageError("member component IDs must be a JSON list")
        result = cls(
            scenario_id=raw["scenario_id"],
            hypothesis_id=raw["hypothesis_id"],
            hypothesis_digest=raw["hypothesis_digest"],
            union_mask_digest=raw["union_mask_digest"],
            union_area_pixels=raw["union_area_pixels"],
            bbox_pixels=_bbox(raw["bbox_pixels"], "member bbox"),
            source_component_ids=tuple(raw["source_component_ids"]),
            scenario_component_count=raw["scenario_component_count"],
            emergence_gap_pixels=raw["emergence_gap_pixels"],
            nearest_external_gap_pixels=raw["nearest_external_gap_pixels"],
        )
        if result.to_data() != dict(raw):
            raise ObjectLineageError("lineage member is not canonical")
        return result


@dataclass(frozen=True, order=True, slots=True)
class ObjectLineageLink:
    left_scenario_id: str
    left_hypothesis_id: str
    right_scenario_id: str
    right_hypothesis_id: str
    intersection_pixels: int
    union_pixels: int
    mask_iou_ppm: int
    bbox_intersection_pixels: int
    bbox_union_pixels: int
    bbox_iou_ppm: int

    def __post_init__(self) -> None:
        left = (self.left_scenario_id, self.left_hypothesis_id)
        right = (self.right_scenario_id, self.right_hypothesis_id)
        if (
            self.left_scenario_id not in VISUAL_WITNESS_SCENARIO_IDS
            or self.right_scenario_id not in VISUAL_WITNESS_SCENARIO_IDS
            or left >= right
        ):
            raise ObjectLineageError("lineage link endpoints are not canonical")
        for value in (self.left_hypothesis_id, self.right_hypothesis_id):
            if not isinstance(value, str) or _HYPOTHESIS_ID.fullmatch(value) is None:
                raise ObjectLineageError("lineage link hypothesis ID is invalid")
        intersection = _integer(self.intersection_pixels, "mask intersection", minimum=1)
        union = _integer(self.union_pixels, "mask union", minimum=1)
        if intersection > union:
            raise ObjectLineageError("mask intersection exceeds union")
        if self.mask_iou_ppm != intersection * PPM_SCALE // union:
            raise ObjectLineageError("mask IoU differs from exact counts")
        bbox_intersection = _integer(
            self.bbox_intersection_pixels, "bbox intersection", minimum=1
        )
        bbox_union = _integer(self.bbox_union_pixels, "bbox union", minimum=1)
        if bbox_intersection > bbox_union:
            raise ObjectLineageError("bbox intersection exceeds union")
        if self.bbox_iou_ppm != bbox_intersection * PPM_SCALE // bbox_union:
            raise ObjectLineageError("bbox IoU differs from exact counts")
        if self.mask_iou_ppm < MIN_MASK_IOU_PPM:
            raise ObjectLineageError("stored link is below the frozen IoU threshold")

    def to_data(self) -> dict[str, object]:
        return {
            "left_scenario_id": self.left_scenario_id,
            "left_hypothesis_id": self.left_hypothesis_id,
            "right_scenario_id": self.right_scenario_id,
            "right_hypothesis_id": self.right_hypothesis_id,
            "intersection_pixels": self.intersection_pixels,
            "union_pixels": self.union_pixels,
            "mask_iou_ppm": self.mask_iou_ppm,
            "bbox_intersection_pixels": self.bbox_intersection_pixels,
            "bbox_union_pixels": self.bbox_union_pixels,
            "bbox_iou_ppm": self.bbox_iou_ppm,
            "reciprocal_unique_best": True,
        }

    @classmethod
    def from_data(cls, value: object) -> "ObjectLineageLink":
        raw = _exact_fields(
            value,
            {
                "left_scenario_id",
                "left_hypothesis_id",
                "right_scenario_id",
                "right_hypothesis_id",
                "intersection_pixels",
                "union_pixels",
                "mask_iou_ppm",
                "bbox_intersection_pixels",
                "bbox_union_pixels",
                "bbox_iou_ppm",
                "reciprocal_unique_best",
            },
            "lineage link",
        )
        if raw["reciprocal_unique_best"] is not True:
            raise ObjectLineageError("lineage link is not reciprocal unique best")
        kwargs = dict(raw)
        kwargs.pop("reciprocal_unique_best")
        result = cls(**kwargs)
        if result.to_data() != dict(raw):
            raise ObjectLineageError("lineage link is not canonical")
        return result


@dataclass(frozen=True, slots=True)
class ObjectLineage:
    lineage_id: str
    members: tuple[ObjectLineageMember, ...]
    links: tuple[ObjectLineageLink, ...]
    ownership_state: LineageOwnershipState
    eligible_for_aggregation: bool
    minimum_mask_iou_ppm: int
    minimum_bbox_iou_ppm: int

    def __post_init__(self) -> None:
        if not isinstance(self.lineage_id, str) or _LINEAGE_ID.fullmatch(
            self.lineage_id
        ) is None:
            raise ObjectLineageError("lineage ID is not canonical")
        if (
            not isinstance(self.members, tuple)
            or any(not isinstance(item, ObjectLineageMember) for item in self.members)
            or tuple(item.scenario_id for item in self.members)
            != VISUAL_WITNESS_SCENARIO_IDS
        ):
            raise ObjectLineageError("lineage must own one member in every scenario")
        expected_pairs = tuple(itertools.combinations(VISUAL_WITNESS_SCENARIO_IDS, 2))
        actual_pairs = tuple(
            (item.left_scenario_id, item.right_scenario_id) for item in self.links
        )
        if actual_pairs != expected_pairs:
            raise ObjectLineageError("lineage links must cover all scenario pairs")
        by_scenario = {item.scenario_id: item for item in self.members}
        for link in self.links:
            if (
                link.left_hypothesis_id
                != by_scenario[link.left_scenario_id].hypothesis_id
                or link.right_hypothesis_id
                != by_scenario[link.right_scenario_id].hypothesis_id
            ):
                raise ObjectLineageError("lineage link differs from its members")
        if not isinstance(self.ownership_state, LineageOwnershipState):
            raise TypeError("ownership_state must be LineageOwnershipState")
        all_singleton = all(item.is_singleton for item in self.members)
        all_persistent = all(item.persistent_before_external_merge for item in self.members)
        any_whole_union = any(item.is_whole_scene_union for item in self.members)
        expected_state = (
            LineageOwnershipState.SAFE_SINGLETON
            if all_singleton
            else LineageOwnershipState.SAFE_PERSISTENT_UNION
            if all_persistent and not any_whole_union
            else LineageOwnershipState.UNRESOLVED_UNION
        )
        if self.ownership_state is not expected_state:
            raise ObjectLineageError("lineage ownership state differs from gap replay")
        if self.eligible_for_aggregation is not (
            expected_state is not LineageOwnershipState.UNRESOLVED_UNION
        ):
            raise ObjectLineageError("lineage aggregation eligibility differs")
        if self.minimum_mask_iou_ppm != min(item.mask_iou_ppm for item in self.links):
            raise ObjectLineageError("minimum lineage mask IoU differs")
        if self.minimum_bbox_iou_ppm != min(item.bbox_iou_ppm for item in self.links):
            raise ObjectLineageError("minimum lineage bbox IoU differs")

    def to_data(self) -> dict[str, object]:
        return {
            "lineage_id": self.lineage_id,
            "members": [item.to_data() for item in self.members],
            "links": [item.to_data() for item in self.links],
            "ownership_state": self.ownership_state.value,
            "eligible_for_aggregation": self.eligible_for_aggregation,
            "minimum_mask_iou_ppm": self.minimum_mask_iou_ppm,
            "minimum_bbox_iou_ppm": self.minimum_bbox_iou_ppm,
        }

    @classmethod
    def from_data(cls, value: object) -> "ObjectLineage":
        raw = _exact_fields(
            value,
            {
                "lineage_id",
                "members",
                "links",
                "ownership_state",
                "eligible_for_aggregation",
                "minimum_mask_iou_ppm",
                "minimum_bbox_iou_ppm",
            },
            "object lineage",
        )
        if not isinstance(raw["members"], list) or not isinstance(raw["links"], list):
            raise ObjectLineageError("lineage members and links must be JSON lists")
        try:
            state = LineageOwnershipState(raw["ownership_state"])
        except (TypeError, ValueError) as exc:
            raise ObjectLineageError("unknown lineage ownership state") from exc
        result = cls(
            lineage_id=raw["lineage_id"],
            members=tuple(ObjectLineageMember.from_data(item) for item in raw["members"]),
            links=tuple(ObjectLineageLink.from_data(item) for item in raw["links"]),
            ownership_state=state,
            eligible_for_aggregation=raw["eligible_for_aggregation"],
            minimum_mask_iou_ppm=raw["minimum_mask_iou_ppm"],
            minimum_bbox_iou_ppm=raw["minimum_bbox_iou_ppm"],
        )
        if result.to_data() != dict(raw):
            raise ObjectLineageError("object lineage is not canonical")
        return result


@dataclass(frozen=True, slots=True)
class ObjectLineagePacket:
    panel_digest: str
    width_pixels: int
    height_pixels: int
    hypothesis_packet_digest: str
    hypothesis_extractor_artifact_digest: str
    source_digest: str
    extractor_artifact_digest: str
    hypothesis_count: int
    linked_hypothesis_count: int
    unlinked_hypothesis_count: int
    ambiguous_member_target_count: int
    has_unresolved_lineages: bool
    lineages: tuple[ObjectLineage, ...]

    def __post_init__(self) -> None:
        _digest(self.panel_digest, "lineage panel digest")
        _integer(self.width_pixels, "lineage width", minimum=2)
        _integer(self.height_pixels, "lineage height", minimum=2)
        _digest(self.hypothesis_packet_digest, "hypothesis packet digest")
        if self.hypothesis_extractor_artifact_digest != (
            object_hypothesis_extractor_artifact_digest()
        ):
            raise ObjectLineageError("hypothesis extractor identity drifted")
        if self.source_digest != object_lineage_source_digest():
            raise ObjectLineageError("lineage source identity drifted")
        if self.extractor_artifact_digest != object_lineage_artifact_digest():
            raise ObjectLineageError("lineage artifact identity drifted")
        if not isinstance(self.lineages, tuple) or any(
            not isinstance(item, ObjectLineage) for item in self.lineages
        ):
            raise TypeError("lineages must be a typed tuple")
        expected_ids = tuple(
            f"lineage-{index:08d}" for index in range(len(self.lineages))
        )
        if tuple(item.lineage_id for item in self.lineages) != expected_ids:
            raise ObjectLineageError("lineages are not in canonical order")
        member_keys = tuple(
            (member.scenario_id, member.hypothesis_id)
            for lineage in self.lineages
            for member in lineage.members
        )
        if len(member_keys) != len(set(member_keys)):
            raise ObjectLineageError("a hypothesis belongs to multiple lineages")
        hypothesis_count = _integer(self.hypothesis_count, "hypothesis count")
        linked_count = _integer(
            self.linked_hypothesis_count, "linked hypothesis count"
        )
        if linked_count != len(member_keys) or linked_count > hypothesis_count:
            raise ObjectLineageError("linked hypothesis count differs")
        if self.unlinked_hypothesis_count != hypothesis_count - linked_count:
            raise ObjectLineageError("unlinked hypothesis count differs")
        _integer(
            self.ambiguous_member_target_count,
            "ambiguous member-target count",
        )
        eligible_count = sum(item.eligible_for_aggregation for item in self.lineages)
        expected_unresolved = (
            self.ambiguous_member_target_count > 0 or eligible_count == 0
        )
        if self.has_unresolved_lineages is not expected_unresolved:
            raise ObjectLineageError("unresolved-lineage flag differs")

    def to_data(self) -> dict[str, object]:
        return {
            "schema": OBJECT_LINEAGE_PACKET_SCHEMA,
            "algorithm_id": OBJECT_LINEAGE_ALGORITHM_ID,
            "panel_digest": self.panel_digest,
            "width_pixels": self.width_pixels,
            "height_pixels": self.height_pixels,
            "hypothesis_packet_digest": self.hypothesis_packet_digest,
            "hypothesis_extractor_artifact_digest": (
                self.hypothesis_extractor_artifact_digest
            ),
            "source_digest": self.source_digest,
            "extractor_artifact_digest": self.extractor_artifact_digest,
            "hypothesis_count": self.hypothesis_count,
            "linked_hypothesis_count": self.linked_hypothesis_count,
            "unlinked_hypothesis_count": self.unlinked_hypothesis_count,
            "ambiguous_member_target_count": self.ambiguous_member_target_count,
            "has_unresolved_lineages": self.has_unresolved_lineages,
            "lineages": [item.to_data() for item in self.lineages],
            "linkage_policy": _linkage_policy_data(),
            "runtime_authority": _authority_data(),
        }

    @classmethod
    def from_data(cls, value: object) -> "ObjectLineagePacket":
        raw = _exact_fields(
            value,
            {
                "schema",
                "algorithm_id",
                "panel_digest",
                "width_pixels",
                "height_pixels",
                "hypothesis_packet_digest",
                "hypothesis_extractor_artifact_digest",
                "source_digest",
                "extractor_artifact_digest",
                "hypothesis_count",
                "linked_hypothesis_count",
                "unlinked_hypothesis_count",
                "ambiguous_member_target_count",
                "has_unresolved_lineages",
                "lineages",
                "linkage_policy",
                "runtime_authority",
            },
            "object lineage packet",
        )
        if (
            raw["schema"] != OBJECT_LINEAGE_PACKET_SCHEMA
            or raw["algorithm_id"] != OBJECT_LINEAGE_ALGORITHM_ID
            or raw["linkage_policy"] != _linkage_policy_data()
            or raw["runtime_authority"] != _authority_data()
            or not isinstance(raw["lineages"], list)
        ):
            raise ObjectLineageError("unsupported object lineage packet")
        result = cls(
            panel_digest=raw["panel_digest"],
            width_pixels=raw["width_pixels"],
            height_pixels=raw["height_pixels"],
            hypothesis_packet_digest=raw["hypothesis_packet_digest"],
            hypothesis_extractor_artifact_digest=raw[
                "hypothesis_extractor_artifact_digest"
            ],
            source_digest=raw["source_digest"],
            extractor_artifact_digest=raw["extractor_artifact_digest"],
            hypothesis_count=raw["hypothesis_count"],
            linked_hypothesis_count=raw["linked_hypothesis_count"],
            unlinked_hypothesis_count=raw["unlinked_hypothesis_count"],
            ambiguous_member_target_count=raw[
                "ambiguous_member_target_count"
            ],
            has_unresolved_lineages=raw["has_unresolved_lineages"],
            lineages=tuple(ObjectLineage.from_data(item) for item in raw["lineages"]),
        )
        if result.to_data() != dict(raw):
            raise ObjectLineageError("object lineage packet is not canonical")
        return result

    def digest(self) -> str:
        return canonical_digest(self.to_data())


def _combined_cell_state(
    states: Sequence[ObjectFeatureCellState],
) -> ObjectFeatureCellState:
    frozen = tuple(states)
    if len(frozen) != len(VISUAL_WITNESS_SCENARIO_IDS):
        raise ObjectLineageError("lineage feature must cover every scenario")
    if ObjectFeatureCellState.ERROR in frozen:
        return ObjectFeatureCellState.ERROR
    if ObjectFeatureCellState.INDETERMINATE in frozen:
        return ObjectFeatureCellState.INDETERMINATE
    return ObjectFeatureCellState.SCORED


def _lineage_feature_content(
    value: "ObjectLineageFeatureEvidence",
) -> dict[str, object]:
    return {
        "schema": OBJECT_LINEAGE_FEATURE_SCHEMA,
        "feature_id": value.feature_id,
        "state": value.state.value,
        "interval": None if value.interval is None else value.interval.to_data(),
        "member_states": [item.value for item in value.member_states],
        "member_cell_digests": list(value.member_cell_digests),
        "reason": value.reason,
        "error_type": value.error_type,
        "aggregation": "same-lineage-min-lower-max-upper",
    }


@dataclass(frozen=True, slots=True)
class ObjectLineageFeatureEvidence:
    """One full-scenario feature envelope for one stable object lineage."""

    feature_id: str
    state: ObjectFeatureCellState
    interval: IntegerInterval | None
    member_states: tuple[ObjectFeatureCellState, ...]
    member_cell_digests: tuple[str, ...]
    reason: str | None
    error_type: str | None
    feature_digest: str

    def __post_init__(self) -> None:
        if self.feature_id not in OBJECT_FEATURE_IDS:
            raise ObjectLineageError("lineage feature is outside frozen catalog")
        if not isinstance(self.state, ObjectFeatureCellState):
            raise TypeError("lineage feature state has wrong type")
        if (
            not isinstance(self.member_states, tuple)
            or any(not isinstance(item, ObjectFeatureCellState) for item in self.member_states)
            or self.state is not _combined_cell_state(self.member_states)
        ):
            raise ObjectLineageError("lineage member feature states differ")
        if (
            not isinstance(self.member_cell_digests, tuple)
            or len(self.member_cell_digests) != len(self.member_states)
        ):
            raise ObjectLineageError("lineage member cell commitments differ")
        for digest in self.member_cell_digests:
            _digest(digest, "member feature cell digest")
        if self.state is ObjectFeatureCellState.SCORED:
            if (
                not isinstance(self.interval, IntegerInterval)
                or self.reason is not None
                or self.error_type is not None
            ):
                raise ObjectLineageError("scored lineage feature differs")
            maximum = next(
                item.maximum
                for item in OBJECT_FEATURE_CATALOG
                if item.feature_id == self.feature_id
            )
            if maximum is not None and self.interval.upper > maximum:
                raise ObjectLineageError("lineage feature interval exceeds catalog")
        elif self.state is ObjectFeatureCellState.INDETERMINATE:
            if (
                self.interval is not None
                or self.reason != "lineage_member_indeterminate"
                or self.error_type is not None
            ):
                raise ObjectLineageError("indeterminate lineage feature differs")
        elif (
            self.interval is not None
            or self.reason != "lineage_member_error"
            or self.error_type != "ObjectFeatureCellError"
        ):
            raise ObjectLineageError("error lineage feature differs")
        _digest(self.feature_digest, "lineage feature digest")
        if self.feature_digest != canonical_digest(_lineage_feature_content(self)):
            raise ObjectLineageError("lineage feature digest differs")

    @classmethod
    def create(
        cls, feature_id: str, cells: Sequence[ObjectFeatureCell]
    ) -> "ObjectLineageFeatureEvidence":
        frozen = tuple(cells)
        if (
            len(frozen) != len(VISUAL_WITNESS_SCENARIO_IDS)
            or any(item.feature_id != feature_id for item in frozen)
        ):
            raise ObjectLineageError("lineage feature cells do not align")
        states = tuple(item.state for item in frozen)
        state = _combined_cell_state(states)
        interval = None
        reason = None
        error_type = None
        if state is ObjectFeatureCellState.SCORED:
            intervals = tuple(item.interval for item in frozen)
            if any(not isinstance(item, IntegerInterval) for item in intervals):
                raise ObjectLineageError("scored member lacks an interval")
            interval = IntegerInterval(
                min(item.lower for item in intervals if item is not None),
                max(item.upper for item in intervals if item is not None),
            )
        elif state is ObjectFeatureCellState.INDETERMINATE:
            reason = "lineage_member_indeterminate"
        else:
            reason = "lineage_member_error"
            error_type = "ObjectFeatureCellError"
        values: dict[str, object] = {
            "feature_id": feature_id,
            "state": state,
            "interval": interval,
            "member_states": states,
            "member_cell_digests": tuple(
                canonical_digest(item.to_data()) for item in frozen
            ),
            "reason": reason,
            "error_type": error_type,
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(
            **values,  # type: ignore[arg-type]
            feature_digest=canonical_digest(_lineage_feature_content(provisional)),
        )

    def to_data(self) -> dict[str, object]:
        return {**_lineage_feature_content(self), "feature_digest": self.feature_digest}

    @classmethod
    def from_data(cls, value: object) -> "ObjectLineageFeatureEvidence":
        raw = _exact_fields(
            value,
            {
                "schema", "feature_id", "state", "interval", "member_states",
                "member_cell_digests", "reason", "error_type", "aggregation",
                "feature_digest",
            },
            "lineage feature evidence",
        )
        if (
            raw["schema"] != OBJECT_LINEAGE_FEATURE_SCHEMA
            or raw["aggregation"] != "same-lineage-min-lower-max-upper"
            or not isinstance(raw["member_states"], list)
            or not isinstance(raw["member_cell_digests"], list)
        ):
            raise ObjectLineageError("lineage feature policy differs")
        try:
            state = ObjectFeatureCellState(raw["state"])
            member_states = tuple(
                ObjectFeatureCellState(item) for item in raw["member_states"]
            )
        except (TypeError, ValueError) as exc:
            raise ObjectLineageError("lineage feature state is unknown") from exc
        result = cls(
            feature_id=raw["feature_id"],
            state=state,
            interval=(
                None
                if raw["interval"] is None
                else IntegerInterval.from_data(raw["interval"])
            ),
            member_states=member_states,
            member_cell_digests=tuple(raw["member_cell_digests"]),
            reason=raw["reason"],
            error_type=raw["error_type"],
            feature_digest=raw["feature_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectLineageError("lineage feature is not canonical")
        return result


def _lineage_observation_content(
    value: "ObjectLineageObservation",
) -> dict[str, object]:
    return {
        "schema": OBJECT_LINEAGE_OBSERVATION_SCHEMA,
        "lineage_id": value.lineage_id,
        "member_hypothesis_ids": list(value.member_hypothesis_ids),
        "geometry_digest": value.geometry_digest,
        "feature_catalog_digest": OBJECT_FEATURE_CATALOG_DIGEST,
        "features": [item.to_data() for item in value.features],
        "same_object_across_all_scenarios": True,
        "eligible_ownership": True,
    }


@dataclass(frozen=True, slots=True)
class ObjectLineageObservation:
    lineage_id: str
    member_hypothesis_ids: tuple[str, ...]
    geometry_digest: str
    features: tuple[ObjectLineageFeatureEvidence, ...]
    observation_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.lineage_id, str) or _LINEAGE_ID.fullmatch(self.lineage_id) is None:
            raise ObjectLineageError("observation lineage ID is invalid")
        if (
            not isinstance(self.member_hypothesis_ids, tuple)
            or len(self.member_hypothesis_ids) != len(VISUAL_WITNESS_SCENARIO_IDS)
            or any(_HYPOTHESIS_ID.fullmatch(item) is None for item in self.member_hypothesis_ids)
        ):
            raise ObjectLineageError("lineage observation members differ")
        _digest(self.geometry_digest, "lineage geometry digest")
        if (
            not isinstance(self.features, tuple)
            or any(not isinstance(item, ObjectLineageFeatureEvidence) for item in self.features)
            or tuple(item.feature_id for item in self.features) != OBJECT_FEATURE_IDS
        ):
            raise ObjectLineageError("lineage observation must exhaust feature catalog")
        _digest(self.observation_digest, "lineage observation digest")
        if self.observation_digest != canonical_digest(_lineage_observation_content(self)):
            raise ObjectLineageError("lineage observation digest differs")

    @classmethod
    def create(
        cls,
        lineage: ObjectLineage,
        features: Sequence[ObjectLineageFeatureEvidence],
    ) -> "ObjectLineageObservation":
        if not isinstance(lineage, ObjectLineage) or not lineage.eligible_for_aggregation:
            raise ObjectLineageError("ineligible geometry cannot create an observation")
        frozen = tuple(features)
        values: dict[str, object] = {
            "lineage_id": lineage.lineage_id,
            "member_hypothesis_ids": tuple(
                item.hypothesis_id for item in lineage.members
            ),
            "geometry_digest": canonical_digest(lineage.to_data()),
            "features": frozen,
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(
            **values,  # type: ignore[arg-type]
            observation_digest=canonical_digest(_lineage_observation_content(provisional)),
        )

    def to_data(self) -> dict[str, object]:
        return {**_lineage_observation_content(self), "observation_digest": self.observation_digest}

    @classmethod
    def from_data(cls, value: object) -> "ObjectLineageObservation":
        raw = _exact_fields(
            value,
            {
                "schema", "lineage_id", "member_hypothesis_ids", "geometry_digest",
                "feature_catalog_digest", "features",
                "same_object_across_all_scenarios", "eligible_ownership",
                "observation_digest",
            },
            "lineage observation",
        )
        if (
            raw["schema"] != OBJECT_LINEAGE_OBSERVATION_SCHEMA
            or raw["feature_catalog_digest"] != OBJECT_FEATURE_CATALOG_DIGEST
            or raw["same_object_across_all_scenarios"] is not True
            or raw["eligible_ownership"] is not True
            or not isinstance(raw["member_hypothesis_ids"], list)
            or not isinstance(raw["features"], list)
        ):
            raise ObjectLineageError("lineage observation policy differs")
        result = cls(
            lineage_id=raw["lineage_id"],
            member_hypothesis_ids=tuple(raw["member_hypothesis_ids"]),
            geometry_digest=raw["geometry_digest"],
            features=tuple(
                ObjectLineageFeatureEvidence.from_data(item)
                for item in raw["features"]
            ),
            observation_digest=raw["observation_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectLineageError("lineage observation is not canonical")
        return result


def _aggregation_content(
    value: "ObjectLineageObservationAggregation",
) -> dict[str, object]:
    return {
        "schema": OBJECT_LINEAGE_AGGREGATION_SCHEMA,
        "algorithm_id": OBJECT_LINEAGE_AGGREGATION_ID,
        "lineage_packet_digest": value.lineage_packet_digest,
        "panel_digest": value.panel_digest,
        "local_packet_digests": list(value.local_packet_digests),
        "lineages": [item.to_data() for item in value.lineages],
        "excluded_lineage_ids": list(value.excluded_lineage_ids),
        "unresolved_lineage_possible": value.unresolved_lineage_possible,
        "unlinked_hypothesis_count": value.unlinked_hypothesis_count,
        "ambiguous_member_target_count": value.ambiguous_member_target_count,
        "aggregation_policy": _aggregation_policy_data(),
        "runtime_authority": _authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectLineageObservationAggregation:
    lineage_packet_digest: str
    panel_digest: str
    local_packet_digests: tuple[str, ...]
    lineages: tuple[ObjectLineageObservation, ...]
    excluded_lineage_ids: tuple[str, ...]
    unresolved_lineage_possible: bool
    unlinked_hypothesis_count: int
    ambiguous_member_target_count: int
    aggregation_digest: str

    def __post_init__(self) -> None:
        _digest(self.lineage_packet_digest, "lineage packet digest")
        _digest(self.panel_digest, "aggregation panel digest")
        if (
            not isinstance(self.local_packet_digests, tuple)
            or len(self.local_packet_digests) != len(VISUAL_WITNESS_SCENARIO_IDS)
        ):
            raise ObjectLineageError("aggregation packet manifest differs")
        for item in self.local_packet_digests:
            _digest(item, "local observation packet digest")
        if (
            not isinstance(self.lineages, tuple)
            or any(not isinstance(item, ObjectLineageObservation) for item in self.lineages)
            or tuple(item.lineage_id for item in self.lineages)
            != tuple(sorted(item.lineage_id for item in self.lineages))
            or not isinstance(self.excluded_lineage_ids, tuple)
            or self.excluded_lineage_ids != tuple(sorted(set(self.excluded_lineage_ids)))
            or set(self.excluded_lineage_ids) & {item.lineage_id for item in self.lineages}
        ):
            raise ObjectLineageError("aggregation lineage inventory differs")
        if type(self.unresolved_lineage_possible) is not bool:
            raise TypeError("aggregation unresolved flag must be bool")
        _integer(self.unlinked_hypothesis_count, "unlinked hypothesis count")
        _integer(self.ambiguous_member_target_count, "ambiguous target count")
        if self.unresolved_lineage_possible is not (
            self.ambiguous_member_target_count > 0 or not self.lineages
        ):
            raise ObjectLineageError("aggregation unresolved flag differs")
        _digest(self.aggregation_digest, "aggregation digest")
        if self.aggregation_digest != canonical_digest(_aggregation_content(self)):
            raise ObjectLineageError("aggregation digest differs")

    def to_data(self) -> dict[str, object]:
        return {**_aggregation_content(self), "aggregation_digest": self.aggregation_digest}

    @classmethod
    def from_data(cls, value: object) -> "ObjectLineageObservationAggregation":
        raw = _exact_fields(
            value,
            {
                "schema", "algorithm_id", "lineage_packet_digest", "panel_digest",
                "local_packet_digests", "lineages", "excluded_lineage_ids",
                "unresolved_lineage_possible", "unlinked_hypothesis_count",
                "ambiguous_member_target_count", "aggregation_policy",
                "runtime_authority", "aggregation_digest",
            },
            "lineage observation aggregation",
        )
        if (
            raw["schema"] != OBJECT_LINEAGE_AGGREGATION_SCHEMA
            or raw["algorithm_id"] != OBJECT_LINEAGE_AGGREGATION_ID
            or raw["aggregation_policy"] != _aggregation_policy_data()
            or raw["runtime_authority"] != _authority_data()
            or not isinstance(raw["local_packet_digests"], list)
            or not isinstance(raw["lineages"], list)
            or not isinstance(raw["excluded_lineage_ids"], list)
        ):
            raise ObjectLineageError("aggregation policy differs")
        result = cls(
            lineage_packet_digest=raw["lineage_packet_digest"],
            panel_digest=raw["panel_digest"],
            local_packet_digests=tuple(raw["local_packet_digests"]),
            lineages=tuple(ObjectLineageObservation.from_data(item) for item in raw["lineages"]),
            excluded_lineage_ids=tuple(raw["excluded_lineage_ids"]),
            unresolved_lineage_possible=raw["unresolved_lineage_possible"],
            unlinked_hypothesis_count=raw["unlinked_hypothesis_count"],
            ambiguous_member_target_count=raw["ambiguous_member_target_count"],
            aggregation_digest=raw["aggregation_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectLineageError("aggregation is not canonical")
        return result


@dataclass(frozen=True, slots=True)
class _Candidate:
    hypothesis: ObjectHypothesis
    mask: np.ndarray
    scenario_component_count: int
    nearest_external_gap_pixels: int | None

    @property
    def key(self) -> tuple[str, str]:
        return (self.hypothesis.scenario_id, self.hypothesis.hypothesis_id)


@dataclass(frozen=True, slots=True)
class _PairMetric:
    left: _Candidate
    right: _Candidate
    intersection_pixels: int
    union_pixels: int
    mask_iou_ppm: int
    bbox_intersection_pixels: int
    bbox_union_pixels: int
    bbox_iou_ppm: int

    @property
    def qualifies(self) -> bool:
        return (
            self.mask_iou_ppm >= MIN_MASK_IOU_PPM
            and self.bbox_intersection_pixels > 0
        )

    @property
    def score(self) -> tuple[int, int, int, int]:
        area_delta = abs(
            self.left.hypothesis.union_area_pixels
            - self.right.hypothesis.union_area_pixels
        )
        return (
            self.mask_iou_ppm,
            self.bbox_iou_ppm,
            self.intersection_pixels,
            -area_delta,
        )

    def to_link(self) -> ObjectLineageLink:
        left, right = self.left, self.right
        if left.key > right.key:
            left, right = right, left
        return ObjectLineageLink(
            left_scenario_id=left.key[0],
            left_hypothesis_id=left.key[1],
            right_scenario_id=right.key[0],
            right_hypothesis_id=right.key[1],
            intersection_pixels=self.intersection_pixels,
            union_pixels=self.union_pixels,
            mask_iou_ppm=self.mask_iou_ppm,
            bbox_intersection_pixels=self.bbox_intersection_pixels,
            bbox_union_pixels=self.bbox_union_pixels,
            bbox_iou_ppm=self.bbox_iou_ppm,
        )


def _bbox_overlap_counts(
    left: tuple[int, int, int, int], right: tuple[int, int, int, int]
) -> tuple[int, int]:
    lx0, ly0, lx1, ly1 = left
    rx0, ry0, rx1, ry1 = right
    width = max(0, min(lx1, rx1) - max(lx0, rx0))
    height = max(0, min(ly1, ry1) - max(ly0, ry0))
    intersection = width * height
    left_area = (lx1 - lx0) * (ly1 - ly0)
    right_area = (rx1 - rx0) * (ry1 - ry0)
    return intersection, left_area + right_area - intersection


def _pair_metric(left: _Candidate, right: _Candidate) -> _PairMetric:
    if left.key >= right.key:
        raise ObjectLineageError("pair metric endpoints are not ordered")
    intersection = int(np.count_nonzero(left.mask & right.mask))
    union = int(np.count_nonzero(left.mask | right.mask))
    bbox_intersection, bbox_union = _bbox_overlap_counts(
        left.hypothesis.bbox_pixels, right.hypothesis.bbox_pixels
    )
    return _PairMetric(
        left=left,
        right=right,
        intersection_pixels=intersection,
        union_pixels=union,
        mask_iou_ppm=intersection * PPM_SCALE // union,
        bbox_intersection_pixels=bbox_intersection,
        bbox_union_pixels=bbox_union,
        bbox_iou_ppm=(
            bbox_intersection * PPM_SCALE // bbox_union if bbox_union else 0
        ),
    )


def _nearest_external_gap(
    hypothesis: ObjectHypothesis, scenario_hypotheses: Sequence[ObjectHypothesis]
) -> int | None:
    owned = frozenset(hypothesis.source_component_ids)
    supersets = [
        item.emergence_gap_pixels
        for item in scenario_hypotheses
        if owned < frozenset(item.source_component_ids)
    ]
    return min(supersets) if supersets else None


def _replay_candidates(
    png_bytes: bytes, packet: ObjectHypothesisPacket
) -> dict[str, tuple[_Candidate, ...]]:
    visual_packet = _visual.extract_visual_witnesses(png_bytes)
    if visual_packet.digest() != packet.visual_witness_packet_digest:
        raise ObjectLineageError("visual witness packet binding differs")
    strength = _visual._decode_png(png_bytes)
    result: dict[str, tuple[_Candidate, ...]] = {}
    for visual_scenario, hypothesis_scenario in zip(
        visual_packet.scenarios, packet.scenarios, strict=True
    ):
        component_masks = _hypotheses._component_masks(strength, visual_scenario)
        by_component = {
            component.component_id: mask
            for component, mask in zip(
                visual_scenario.components, component_masks, strict=True
            )
        }
        candidates: list[_Candidate] = []
        for hypothesis in hypothesis_scenario.hypotheses:
            union = np.zeros_like(strength, dtype=bool)
            for component_id in hypothesis.source_component_ids:
                union |= by_component[component_id]
            if (
                _visual._mask_digest(union) != hypothesis.union_mask_digest
                or int(np.count_nonzero(union)) != hypothesis.union_area_pixels
                or _visual._bbox(union) != hypothesis.bbox_pixels
            ):
                raise ObjectLineageError("hypothesis exact-mask replay differs")
            candidates.append(
                _Candidate(
                    hypothesis=hypothesis,
                    mask=np.ascontiguousarray(union),
                    scenario_component_count=len(visual_scenario.components),
                    nearest_external_gap_pixels=_nearest_external_gap(
                        hypothesis, hypothesis_scenario.hypotheses
                    ),
                )
            )
        result[visual_scenario.scenario_id] = tuple(candidates)
    return result


def _reciprocal_unique_links(
    by_scenario: Mapping[str, tuple[_Candidate, ...]],
) -> tuple[
    dict[tuple[tuple[str, str], tuple[str, str]], _PairMetric], int
]:
    reciprocal: dict[tuple[tuple[str, str], tuple[str, str]], _PairMetric] = {}
    ambiguous_count = 0
    for left_scenario, right_scenario in itertools.combinations(
        VISUAL_WITNESS_SCENARIO_IDS, 2
    ):
        metrics: dict[tuple[tuple[str, str], tuple[str, str]], _PairMetric] = {}
        for left in by_scenario[left_scenario]:
            for right in by_scenario[right_scenario]:
                metric = _pair_metric(left, right)
                if metric.qualifies:
                    metrics[(left.key, right.key)] = metric
        left_best: dict[tuple[str, str], tuple[str, str]] = {}
        right_best: dict[tuple[str, str], tuple[str, str]] = {}
        for source, source_is_left in (
            (by_scenario[left_scenario], True),
            (by_scenario[right_scenario], False),
        ):
            for candidate in source:
                rows = [
                    metric
                    for (left_key, right_key), metric in metrics.items()
                    if (left_key if source_is_left else right_key) == candidate.key
                ]
                if not rows:
                    continue
                top_score = max(item.score for item in rows)
                winners = [item for item in rows if item.score == top_score]
                if len(winners) != 1:
                    ambiguous_count += 1
                    continue
                winner = winners[0]
                target = winner.right.key if source_is_left else winner.left.key
                (left_best if source_is_left else right_best)[candidate.key] = target
        for key, metric in metrics.items():
            left_key, right_key = key
            if (
                left_best.get(left_key) == right_key
                and right_best.get(right_key) == left_key
            ):
                reciprocal[key] = metric
    return reciprocal, ambiguous_count


def _member(candidate: _Candidate) -> ObjectLineageMember:
    hypothesis = candidate.hypothesis
    return ObjectLineageMember(
        scenario_id=hypothesis.scenario_id,
        hypothesis_id=hypothesis.hypothesis_id,
        hypothesis_digest=hypothesis.digest(),
        union_mask_digest=hypothesis.union_mask_digest,
        union_area_pixels=hypothesis.union_area_pixels,
        bbox_pixels=hypothesis.bbox_pixels,
        source_component_ids=hypothesis.source_component_ids,
        scenario_component_count=candidate.scenario_component_count,
        emergence_gap_pixels=hypothesis.emergence_gap_pixels,
        nearest_external_gap_pixels=candidate.nearest_external_gap_pixels,
    )


def _build_lineage_packet(
    png_bytes: bytes, hypothesis_packet: ObjectHypothesisPacket
) -> ObjectLineagePacket:
    verify_object_hypothesis_packet(hypothesis_packet, png_bytes)
    by_scenario = _replay_candidates(png_bytes, hypothesis_packet)
    reciprocal, ambiguous_count = _reciprocal_unique_links(by_scenario)
    first, second, third = VISUAL_WITNESS_SCENARIO_IDS
    raw: list[tuple[_Candidate, _Candidate, _Candidate, tuple[_PairMetric, ...]]] = []
    for left in by_scenario[first]:
        second_rows = [
            (key, metric)
            for key, metric in reciprocal.items()
            if key[0] == left.key and key[1][0] == second
        ]
        third_rows = [
            (key, metric)
            for key, metric in reciprocal.items()
            if key[0] == left.key and key[1][0] == third
        ]
        if len(second_rows) != 1 or len(third_rows) != 1:
            continue
        second_key, first_second = second_rows[0]
        third_key, first_third = third_rows[0]
        middle = next(item for item in by_scenario[second] if item.key == second_key[1])
        right = next(item for item in by_scenario[third] if item.key == third_key[1])
        middle_right_key = (middle.key, right.key)
        middle_right = reciprocal.get(middle_right_key)
        if middle_right is None:
            continue
        raw.append((left, middle, right, (first_second, first_third, middle_right)))
    raw.sort(key=lambda row: tuple(item.key for item in row[:3]))
    lineages: list[ObjectLineage] = []
    for index, (left, middle, right, metrics) in enumerate(raw):
        members = tuple(_member(item) for item in (left, middle, right))
        links = tuple(
            sorted((item.to_link() for item in metrics), key=lambda item: (
                item.left_scenario_id, item.right_scenario_id
            ))
        )
        all_singleton = all(item.is_singleton for item in members)
        all_persistent = all(item.persistent_before_external_merge for item in members)
        any_whole = any(item.is_whole_scene_union for item in members)
        ownership = (
            LineageOwnershipState.SAFE_SINGLETON
            if all_singleton
            else LineageOwnershipState.SAFE_PERSISTENT_UNION
            if all_persistent and not any_whole
            else LineageOwnershipState.UNRESOLVED_UNION
        )
        lineages.append(
            ObjectLineage(
                lineage_id=f"lineage-{index:08d}",
                members=members,
                links=links,
                ownership_state=ownership,
                eligible_for_aggregation=(
                    ownership is not LineageOwnershipState.UNRESOLVED_UNION
                ),
                minimum_mask_iou_ppm=min(item.mask_iou_ppm for item in links),
                minimum_bbox_iou_ppm=min(item.bbox_iou_ppm for item in links),
            )
        )
    hypothesis_count = sum(len(item) for item in by_scenario.values())
    linked_count = len(lineages) * len(VISUAL_WITNESS_SCENARIO_IDS)
    frozen = tuple(lineages)
    return ObjectLineagePacket(
        panel_digest=hypothesis_packet.panel_digest,
        width_pixels=hypothesis_packet.width_pixels,
        height_pixels=hypothesis_packet.height_pixels,
        hypothesis_packet_digest=hypothesis_packet.digest(),
        hypothesis_extractor_artifact_digest=(
            hypothesis_packet.extractor_artifact_digest
        ),
        source_digest=object_lineage_source_digest(),
        extractor_artifact_digest=object_lineage_artifact_digest(),
        hypothesis_count=hypothesis_count,
        linked_hypothesis_count=linked_count,
        unlinked_hypothesis_count=hypothesis_count - linked_count,
        ambiguous_member_target_count=ambiguous_count,
        has_unresolved_lineages=(
            ambiguous_count > 0
            or not any(item.eligible_for_aggregation for item in frozen)
        ),
        lineages=frozen,
    )


def extract_object_lineage_packet(
    png_bytes: bytes,
    hypothesis_packet: ObjectHypothesisPacket | None = None,
) -> ObjectLineagePacket:
    """Extract deterministic candidate lineages from exact panel pixels."""

    if not isinstance(png_bytes, bytes):
        raise TypeError("lineage input must be exact PNG bytes")
    if hypothesis_packet is None:
        hypothesis_packet = extract_object_hypothesis_packet(png_bytes)
    elif not isinstance(hypothesis_packet, ObjectHypothesisPacket):
        raise TypeError("hypothesis_packet must be ObjectHypothesisPacket or null")
    return _build_lineage_packet(png_bytes, hypothesis_packet)


def verify_object_lineage_packet(
    packet: ObjectLineagePacket, png_bytes: bytes
) -> ObjectLineagePacket:
    """Cold-replay a lineage packet from its claimed exact PNG bytes."""

    if not isinstance(packet, ObjectLineagePacket):
        raise TypeError("packet must be ObjectLineagePacket")
    if not isinstance(png_bytes, bytes):
        raise TypeError("png_bytes must be exact bytes")
    decoded = ObjectLineagePacket.from_data(packet.to_data())
    expected = extract_object_lineage_packet(png_bytes)
    if decoded != expected:
        raise ObjectLineageError("lineage packet differs from exact PNG replay")
    return decoded


def _canonical_local_packets(
    lineage_packet: ObjectLineagePacket,
    local_packets: Sequence[ObjectLocalObservationPacket],
) -> tuple[ObjectLocalObservationPacket, ...]:
    """Replay and cross-bind the three profile-blind observation packets."""

    if not isinstance(lineage_packet, ObjectLineagePacket):
        raise TypeError("lineage_packet must be ObjectLineagePacket")
    decoded_lineage = ObjectLineagePacket.from_data(lineage_packet.to_data())
    frozen = tuple(local_packets)
    if (
        len(frozen) != len(VISUAL_WITNESS_SCENARIO_IDS)
        or any(not isinstance(item, ObjectLocalObservationPacket) for item in frozen)
    ):
        raise ObjectLineageError(
            "local packets must contain one typed packet per frozen scenario"
        )
    decoded = tuple(
        ObjectLocalObservationPacket.from_data(item.to_data()) for item in frozen
    )
    if tuple(item.scenario_id for item in decoded) != VISUAL_WITNESS_SCENARIO_IDS:
        raise ObjectLineageError("local packet scenarios differ from frozen order")
    for packet in decoded:
        if packet.panel_digest != decoded_lineage.panel_digest:
            raise ObjectLineageError("local packet panel binding differs")
        if (
            packet.hypothesis_catalog_digest
            != decoded_lineage.hypothesis_packet_digest
        ):
            raise ObjectLineageError("local packet hypothesis binding differs")
    shared_fields = (
        "visual_witness_packet_digest",
        "hypothesis_catalog_digest",
        "feature_protocol_digest",
        "feature_model_id",
        "feature_receipt_digest",
        "feature_payload_digest",
    )
    for field in shared_fields:
        if len({getattr(item, field) for item in decoded}) != 1:
            raise ObjectLineageError(
                f"local packet {field} differs across scenarios"
            )

    packet_by_scenario = {item.scenario_id: item for item in decoded}
    for lineage in decoded_lineage.lineages:
        for member in lineage.members:
            packet = packet_by_scenario[member.scenario_id]
            bindings = {
                item.hypothesis_id: item for item in packet.hypotheses
            }
            binding = bindings.get(member.hypothesis_id)
            if binding is None:
                raise ObjectLineageError(
                    "lineage member is absent from its local observation packet"
                )
            if (
                binding.union_mask_digest != member.union_mask_digest
                or binding.union_bbox != member.bbox_pixels
                or binding.source_component_ids != member.source_component_ids
            ):
                raise ObjectLineageError(
                    "lineage member geometry differs from observation binding"
                )
    return decoded


def aggregate_lineage_observations(
    lineage_packet: ObjectLineagePacket,
    local_packets: Sequence[ObjectLocalObservationPacket],
) -> ObjectLineageObservationAggregation:
    """Aggregate feature cells without ever crossing an object lineage.

    Each returned row refers to one reciprocal-stable object across all three
    segmentation scenarios. No maximum or existential reduction is performed
    here; that belongs to the later closed predicate evaluator.
    """

    if not isinstance(lineage_packet, ObjectLineagePacket):
        raise TypeError("lineage_packet must be ObjectLineagePacket")
    lineage_packet = ObjectLineagePacket.from_data(lineage_packet.to_data())
    packets = _canonical_local_packets(lineage_packet, local_packets)
    cells_by_scenario = {
        packet.scenario_id: {
            (cell.hypothesis_id, cell.feature_id): cell
            for cell in packet.cells
        }
        for packet in packets
    }

    observations: list[ObjectLineageObservation] = []
    excluded: list[str] = []
    for lineage in lineage_packet.lineages:
        if not lineage.eligible_for_aggregation:
            excluded.append(lineage.lineage_id)
            continue
        member_by_scenario = {
            item.scenario_id: item for item in lineage.members
        }
        feature_rows = tuple(
            ObjectLineageFeatureEvidence.create(
                feature_id,
                tuple(
                    cells_by_scenario[scenario_id][
                        (
                            member_by_scenario[scenario_id].hypothesis_id,
                            feature_id,
                        )
                    ]
                    for scenario_id in VISUAL_WITNESS_SCENARIO_IDS
                ),
            )
            for feature_id in OBJECT_FEATURE_IDS
        )
        observations.append(ObjectLineageObservation.create(lineage, feature_rows))

    values: dict[str, object] = {
        "lineage_packet_digest": lineage_packet.digest(),
        "panel_digest": lineage_packet.panel_digest,
        "local_packet_digests": tuple(item.packet_digest for item in packets),
        "lineages": tuple(observations),
        "excluded_lineage_ids": tuple(excluded),
        "unresolved_lineage_possible": lineage_packet.has_unresolved_lineages,
        "unlinked_hypothesis_count": lineage_packet.unlinked_hypothesis_count,
        "ambiguous_member_target_count": (
            lineage_packet.ambiguous_member_target_count
        ),
    }
    provisional = object.__new__(ObjectLineageObservationAggregation)
    for name, value in values.items():
        object.__setattr__(provisional, name, value)
    return ObjectLineageObservationAggregation(
        **values,  # type: ignore[arg-type]
        aggregation_digest=canonical_digest(_aggregation_content(provisional)),
    )


def verify_object_lineage_observation_aggregation(
    aggregation: ObjectLineageObservationAggregation,
    lineage_packet: ObjectLineagePacket,
    local_packets: Sequence[ObjectLocalObservationPacket],
) -> ObjectLineageObservationAggregation:
    """Model-free replay of an aggregation against exact committed inputs."""

    if not isinstance(aggregation, ObjectLineageObservationAggregation):
        raise TypeError(
            "aggregation must be ObjectLineageObservationAggregation"
        )
    decoded = ObjectLineageObservationAggregation.from_data(
        aggregation.to_data()
    )
    expected = aggregate_lineage_observations(lineage_packet, local_packets)
    if decoded != expected:
        raise ObjectLineageError(
            "lineage observation aggregation differs from exact replay"
        )
    return decoded


def object_scene_evidence_from_lineage_aggregation(
    scene_id: str,
    aggregation: ObjectLineageObservationAggregation,
):
    """Convert a verified aggregation to the closed Python version-space IR."""

    if not isinstance(aggregation, ObjectLineageObservationAggregation):
        raise TypeError(
            "aggregation must be ObjectLineageObservationAggregation"
        )
    aggregation = ObjectLineageObservationAggregation.from_data(
        aggregation.to_data()
    )
    # Imported only at the boundary to keep geometry independent of synthesis.
    from bongard.prototype_object_version_space import (
        ObjectSceneEvidence,
        ObjectSceneFeatureValue,
        ObjectStableLineageEvidence,
    )

    lineages = []
    for lineage in aggregation.lineages:
        values = []
        for feature in lineage.features:
            if feature.state is ObjectFeatureCellState.SCORED:
                values.append(
                    ObjectSceneFeatureValue(
                        feature.feature_id,
                        Disposition.PRESENT,
                        feature.interval,
                    )
                )
            elif feature.state is ObjectFeatureCellState.INDETERMINATE:
                values.append(
                    ObjectSceneFeatureValue(
                        feature.feature_id,
                        Disposition.INDETERMINATE,
                        None,
                        reason=feature.reason,
                    )
                )
            else:
                values.append(
                    ObjectSceneFeatureValue(
                        feature.feature_id,
                        Disposition.ERROR,
                        None,
                        reason=feature.reason,
                        error_type=feature.error_type,
                    )
                )
        lineages.append(
            ObjectStableLineageEvidence.create(lineage.lineage_id, values)
        )
    return ObjectSceneEvidence.create(
        scene_id,
        object_lineage_artifact_digest(),
        lineages,
        unresolved_lineage_possible=aggregation.unresolved_lineage_possible,
    )


__all__ = (
    "MIN_MASK_IOU_PPM",
    "OBJECT_LINEAGE_AGGREGATION_ID",
    "OBJECT_LINEAGE_ALGORITHM_ID",
    "OBJECT_LINEAGE_PACKET_SCHEMA",
    "LineageOwnershipState",
    "ObjectLineage",
    "ObjectLineageError",
    "ObjectLineageFeatureEvidence",
    "ObjectLineageLink",
    "ObjectLineageMember",
    "ObjectLineageObservation",
    "ObjectLineageObservationAggregation",
    "ObjectLineagePacket",
    "aggregate_lineage_observations",
    "extract_object_lineage_packet",
    "object_lineage_artifact_digest",
    "object_lineage_source_digest",
    "object_scene_evidence_from_lineage_aggregation",
    "verify_object_lineage_observation_aggregation",
    "verify_object_lineage_packet",
)

"""Typed shared-witness contrast IR and its exact Python rubric.

The model does not emit two unrelated descriptions.  It names one visible
entity kind, one visual axis on that entity, and two alternative values of
that axis.  Python renders both descriptions from the same anchor and axis.
This module contains no model transport, threshold search, polarity rescue,
feature fitting, or Lean dependency.
"""

from __future__ import annotations

from bongard.runtime_source_snapshot import capture_loaded_source, verify_loaded_source


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)

from dataclasses import dataclass
import re
from typing import Any, Mapping

from bongard.canonical import canonical_digest
from bongard.object_bongard_soft_cues import (
    ObjectBongardSoftCue,
    ObjectBongardSoftCueError,
    ObjectBongardSoftCuePair,
)
from bongard.python_predicate_authority import PYTHON_PREDICATE_AUTHORITY_ID


SHARED_WITNESS_CONTRAST_SCHEMA = "gkm.bongard-shared-witness-contrast.v1"
SHARED_WITNESS_RUBRIC_SPEC_SCHEMA = "gkm.bongard-shared-witness-rubric-spec.v1"
SHARED_WITNESS_IR_ID = "bongard.shared-witness-contrast/single-entity-single-axis-v1"
SHARED_WITNESS_RENDERER_ID = "bongard.shared-witness-renderer/exact-paired-description-v1"
SHARED_WITNESS_RUBRIC_ID = "bongard.shared-witness-rubric/decomposed-endpoint-evidence-v1"

_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_COMPONENT = re.compile(r"[A-Za-z][A-Za-z' -]*\Z")
_FORBIDDEN_ROLE_OR_LOGIC = re.compile(
    r"\b(?:"
    r"group|class|label|target|foil|positive|negative|reference|example|"
    r"candidate|proposal|description|rule|predicate|formula|threshold|score|"
    r"no|not|none|neither|nor|never|without|lacks?|lacking|absent|absence|"
    r"missing|omits?|omitted|except|different|distinct|unlike|other|versus|than|"
    r"and|or"
    r")\b",
    re.IGNORECASE,
)
_ABSENCE_CODED_ENDPOINT = re.compile(
    r"(?:\b(?:empty|bare|plain|undecorated|unmarked|unfilled|featureless|zero)\b|"
    r"\b[A-Za-z]+less\b)",
    re.IGNORECASE,
)


class ObjectBongardSharedWitnessError(ValueError):
    """A shared-witness IR, rendering, or rubric spec is malformed."""


def _authority_data() -> dict[str, object]:
    return {
        "predicate_authority_id": PYTHON_PREDICATE_AUTHORITY_ID,
        "python_is_canonical_authority": True,
        "model_emits_shared_witness_components_only": True,
        "python_renders_descriptions": True,
        "same_individual_required_within_panel": True,
        "same_physical_individual_across_panels_required": False,
        "same_entity_kind_across_panels_required": True,
        "single_visual_axis_required": True,
        "endpoints_are_alternative_axis_values": True,
        "explicit_or_morphological_absence_endpoint_allowed": False,
        "model_can_choose_operator_threshold_or_polarity": False,
        "negation_allowed": False,
        "polarity_flip_allowed": False,
        "threshold_selection_allowed": False,
        "lean_present": False,
        "lean_required": False,
        "lean_removable": True,
        "lean_defines_identity_or_decision": False,
        "lean_required_for_replay": False,
    }


def _fields(value: object, expected: set[str], label: str) -> Mapping[str, Any]:
    if (
        not isinstance(value, Mapping)
        or any(not isinstance(key, str) for key in value)
        or set(value) != expected
    ):
        raise ObjectBongardSharedWitnessError(f"{label} fields differ")
    return value


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ObjectBongardSharedWitnessError(
            f"{label} must be a raw lowercase SHA-256"
        )
    return value


def object_bongard_shared_witness_source_digest() -> str:
    return verify_loaded_source(__name__, expected_source_sha256=_LOADED_SOURCE_SHA256)


def _component(
    value: object,
    *,
    label: str,
    minimum: int,
    maximum: int,
    maximum_words: int,
    endpoint: bool = False,
) -> str:
    if (
        not isinstance(value, str)
        or not minimum <= len(value) <= maximum
        or value != value.strip()
        or "  " in value
        or _COMPONENT.fullmatch(value) is None
        or not value[0].islower()
        or len(value.split()) > maximum_words
        or _FORBIDDEN_ROLE_OR_LOGIC.search(value) is not None
        or (endpoint and _ABSENCE_CODED_ENDPOINT.search(value) is not None)
    ):
        raise ObjectBongardSharedWitnessError(
            f"{label} violates the atomic positive component grammar"
        )
    return value


def validate_shared_anchor(value: object) -> str:
    return _component(
        value,
        label="shared anchor",
        minimum=3,
        maximum=64,
        maximum_words=8,
    )


def validate_visual_axis(value: object) -> str:
    return _component(
        value,
        label="visual axis",
        minimum=3,
        maximum=72,
        maximum_words=9,
    )


def validate_axis_endpoint(value: object, *, label: str) -> str:
    return _component(
        value,
        label=label,
        minimum=3,
        maximum=88,
        maximum_words=12,
        endpoint=True,
    )


def render_shared_witness_description(
    shared_anchor: str, visual_axis: str, endpoint: str
) -> str:
    """Render one endpoint without allowing the model to vary scope."""

    anchor = validate_shared_anchor(shared_anchor)
    axis = validate_visual_axis(visual_axis)
    value = validate_axis_endpoint(endpoint, label="axis endpoint")
    text = (
        f"The inventoried individual {anchor} is this witness; "
        f"its {axis} appears {value}."
    )
    try:
        return ObjectBongardSoftCue.create(text).text
    except ObjectBongardSoftCueError as exc:
        raise ObjectBongardSharedWitnessError(
            "rendered shared-witness description violates the frozen cue grammar"
        ) from exc


def _contrast_content(value: "ObjectBongardSharedWitnessContrast") -> dict[str, object]:
    return {
        "schema": SHARED_WITNESS_CONTRAST_SCHEMA,
        "ir_id": SHARED_WITNESS_IR_ID,
        "renderer_id": SHARED_WITNESS_RENDERER_ID,
        "implementation_source_sha256": object_bongard_shared_witness_source_digest(),
        "candidate_rank": value.candidate_rank,
        "shared_anchor": value.shared_anchor,
        "visual_axis": value.visual_axis,
        "group_0_endpoint": value.group_0_endpoint,
        "group_1_endpoint": value.group_1_endpoint,
        "rendered_group_0_cue": value.rendered_group_0_cue.to_data(),
        "rendered_group_1_cue": value.rendered_group_1_cue.to_data(),
        "ordered_group_roles": ["group_0", "group_1"],
        "endpoint_relation": "alternative-values-of-one-axis-on-one-individual",
        "panel_witness_policy": "inventory-all-top-level-anchor-instances",
        "witness_cherry_pick_allowed": False,
        "possible_opposite_endpoint_forces_indeterminate": True,
        "reverse_orientation_authorized": False,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectBongardSharedWitnessContrast:
    """One ranked contrast with an indivisible anchor and visual axis."""

    candidate_rank: int
    shared_anchor: str
    visual_axis: str
    group_0_endpoint: str
    group_1_endpoint: str
    rendered_group_0_cue: ObjectBongardSoftCue
    rendered_group_1_cue: ObjectBongardSoftCue
    contrast_digest: str

    def __post_init__(self) -> None:
        if type(self.candidate_rank) is not int or self.candidate_rank not in (0, 1):
            raise ObjectBongardSharedWitnessError(
                "shared-witness candidate rank must be zero or one"
            )
        anchor = validate_shared_anchor(self.shared_anchor)
        axis = validate_visual_axis(self.visual_axis)
        endpoint_0 = validate_axis_endpoint(
            self.group_0_endpoint, label="group zero endpoint"
        )
        endpoint_1 = validate_axis_endpoint(
            self.group_1_endpoint, label="group one endpoint"
        )
        if endpoint_0.casefold() == endpoint_1.casefold():
            raise ObjectBongardSharedWitnessError(
                "shared-witness endpoints must be distinct"
            )
        if not isinstance(self.rendered_group_0_cue, ObjectBongardSoftCue) or not isinstance(
            self.rendered_group_1_cue, ObjectBongardSoftCue
        ):
            raise TypeError("rendered shared-witness cues must be typed")
        expected_0 = ObjectBongardSoftCue.create(
            render_shared_witness_description(anchor, axis, endpoint_0)
        )
        expected_1 = ObjectBongardSoftCue.create(
            render_shared_witness_description(anchor, axis, endpoint_1)
        )
        if (
            self.rendered_group_0_cue != expected_0
            or self.rendered_group_1_cue != expected_1
        ):
            raise ObjectBongardSharedWitnessError(
                "rendered descriptions do not share the exact anchor and axis"
            )
        _digest(self.contrast_digest, "shared-witness contrast digest")
        if self.contrast_digest != canonical_digest(_contrast_content(self)):
            raise ObjectBongardSharedWitnessError(
                "shared-witness contrast digest differs"
            )

    @classmethod
    def create(
        cls,
        candidate_rank: int,
        *,
        shared_anchor: str,
        visual_axis: str,
        group_0_endpoint: str,
        group_1_endpoint: str,
    ) -> "ObjectBongardSharedWitnessContrast":
        anchor = validate_shared_anchor(shared_anchor)
        axis = validate_visual_axis(visual_axis)
        endpoint_0 = validate_axis_endpoint(
            group_0_endpoint, label="group zero endpoint"
        )
        endpoint_1 = validate_axis_endpoint(
            group_1_endpoint, label="group one endpoint"
        )
        values = {
            "candidate_rank": candidate_rank,
            "shared_anchor": anchor,
            "visual_axis": axis,
            "group_0_endpoint": endpoint_0,
            "group_1_endpoint": endpoint_1,
            "rendered_group_0_cue": ObjectBongardSoftCue.create(
                render_shared_witness_description(anchor, axis, endpoint_0)
            ),
            "rendered_group_1_cue": ObjectBongardSoftCue.create(
                render_shared_witness_description(anchor, axis, endpoint_1)
            ),
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(
            **values,
            contrast_digest=canonical_digest(_contrast_content(provisional)),
        )

    @property
    def soft_cue_pair(self) -> ObjectBongardSoftCuePair:
        return ObjectBongardSoftCuePair.create(
            self.candidate_rank,
            self.rendered_group_0_cue,
            self.rendered_group_1_cue,
        )

    def to_data(self) -> dict[str, object]:
        return {**_contrast_content(self), "contrast_digest": self.contrast_digest}

    @classmethod
    def from_data(cls, value: object) -> "ObjectBongardSharedWitnessContrast":
        raw = _fields(
            value,
            {
                "schema",
                "ir_id",
                "renderer_id",
                "implementation_source_sha256",
                "candidate_rank",
                "shared_anchor",
                "visual_axis",
                "group_0_endpoint",
                "group_1_endpoint",
                "rendered_group_0_cue",
                "rendered_group_1_cue",
                "ordered_group_roles",
                "endpoint_relation",
                "panel_witness_policy",
                "witness_cherry_pick_allowed",
                "possible_opposite_endpoint_forces_indeterminate",
                "reverse_orientation_authorized",
                *_authority_data(),
                "contrast_digest",
            },
            "shared-witness contrast",
        )
        if (
            raw["schema"] != SHARED_WITNESS_CONTRAST_SCHEMA
            or raw["ir_id"] != SHARED_WITNESS_IR_ID
            or raw["renderer_id"] != SHARED_WITNESS_RENDERER_ID
            or raw["implementation_source_sha256"]
            != object_bongard_shared_witness_source_digest()
            or raw["ordered_group_roles"] != ["group_0", "group_1"]
            or raw["endpoint_relation"]
            != "alternative-values-of-one-axis-on-one-individual"
            or raw["panel_witness_policy"]
            != "inventory-all-top-level-anchor-instances"
            or raw["witness_cherry_pick_allowed"] is not False
            or raw["possible_opposite_endpoint_forces_indeterminate"] is not True
            or raw["reverse_orientation_authorized"] is not False
            or any(raw[key] != item for key, item in _authority_data().items())
        ):
            raise ObjectBongardSharedWitnessError(
                "shared-witness contrast policy differs"
            )
        result = cls(
            raw["candidate_rank"],
            raw["shared_anchor"],
            raw["visual_axis"],
            raw["group_0_endpoint"],
            raw["group_1_endpoint"],
            ObjectBongardSoftCue.from_data(raw["rendered_group_0_cue"]),
            ObjectBongardSoftCue.from_data(raw["rendered_group_1_cue"]),
            raw["contrast_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectBongardSharedWitnessError(
                "shared-witness contrast is not canonical"
            )
        return result


def render_shared_witness_rubric(
    contrast: ObjectBongardSharedWitnessContrast,
) -> str:
    if not isinstance(contrast, ObjectBongardSharedWitnessContrast):
        raise TypeError("contrast must be ObjectBongardSharedWitnessContrast")
    frozen = ObjectBongardSharedWitnessContrast.from_data(contrast.to_data())
    return (
        f"Inventory every top-level individual {frozen.shared_anchor} in the panel. "
        f"For each inventoried individual, inspect only that same individual's "
        f"{frozen.visual_axis}. "
        f"Description A endpoint is {frozen.group_0_endpoint}. "
        f"Description B endpoint is {frozen.group_1_endpoint}. "
        "Treat A and B only as alternative values of this one visual axis. "
        "Score both endpoints on each same individual. Do not select one favorable "
        "individual or ignore any eligible individual. Any ambiguous eligible "
        "individual or possible opposite endpoint makes the panel indeterminate."
    )


def _spec_content(value: "ObjectBongardSharedWitnessRubricSpec") -> dict[str, object]:
    return {
        "schema": SHARED_WITNESS_RUBRIC_SPEC_SCHEMA,
        "rubric_id": SHARED_WITNESS_RUBRIC_ID,
        "implementation_source_sha256": object_bongard_shared_witness_source_digest(),
        "semantic_artifact_digest": value.semantic_artifact_digest,
        "candidate_rank": value.candidate_rank,
        "contrast": value.contrast.to_data(),
        "rubric": value.rubric,
        "observer_must_persist_witness_locator": True,
        "observer_must_inventory_all_top_level_anchor_instances": True,
        "observer_may_select_one_favorable_witness": False,
        "observer_must_score_endpoints_separately": True,
        "conservative_inventory_aggregation_required": True,
        "possible_opposite_endpoint_forces_indeterminate": True,
        "direct_comparative_judgment_is_canonical_evidence": False,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectBongardSharedWitnessRubricSpec:
    """Content-addressed v2 rubric retaining the complete structured IR."""

    semantic_artifact_digest: str
    candidate_rank: int
    contrast: ObjectBongardSharedWitnessContrast
    rubric: str
    spec_digest: str

    def __post_init__(self) -> None:
        _digest(self.semantic_artifact_digest, "semantic artifact digest")
        if type(self.candidate_rank) is not int or self.candidate_rank not in (0, 1):
            raise ObjectBongardSharedWitnessError("rubric rank must be zero or one")
        if (
            not isinstance(self.contrast, ObjectBongardSharedWitnessContrast)
            or self.contrast.candidate_rank != self.candidate_rank
            or self.rubric != render_shared_witness_rubric(self.contrast)
        ):
            raise ObjectBongardSharedWitnessError(
                "shared-witness rubric does not retain its exact contrast"
            )
        _digest(self.spec_digest, "shared-witness rubric spec digest")
        if self.spec_digest != canonical_digest(_spec_content(self)):
            raise ObjectBongardSharedWitnessError(
                "shared-witness rubric spec digest differs"
            )

    @property
    def target_cue(self) -> ObjectBongardSoftCue:
        return self.contrast.rendered_group_0_cue

    @property
    def foil_cue(self) -> ObjectBongardSoftCue:
        return self.contrast.rendered_group_1_cue

    @classmethod
    def from_contrast(
        cls,
        semantic_artifact_digest: str,
        contrast: ObjectBongardSharedWitnessContrast,
    ) -> "ObjectBongardSharedWitnessRubricSpec":
        digest = _digest(semantic_artifact_digest, "semantic artifact digest")
        frozen = ObjectBongardSharedWitnessContrast.from_data(contrast.to_data())
        values = {
            "semantic_artifact_digest": digest,
            "candidate_rank": frozen.candidate_rank,
            "contrast": frozen,
            "rubric": render_shared_witness_rubric(frozen),
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(
            **values,
            spec_digest=canonical_digest(_spec_content(provisional)),
        )

    def to_data(self) -> dict[str, object]:
        return {**_spec_content(self), "spec_digest": self.spec_digest}

    @classmethod
    def from_data(cls, value: object) -> "ObjectBongardSharedWitnessRubricSpec":
        raw = _fields(
            value,
            {
                "schema",
                "rubric_id",
                "implementation_source_sha256",
                "semantic_artifact_digest",
                "candidate_rank",
                "contrast",
                "rubric",
                "observer_must_persist_witness_locator",
                "observer_must_inventory_all_top_level_anchor_instances",
                "observer_may_select_one_favorable_witness",
                "observer_must_score_endpoints_separately",
                "conservative_inventory_aggregation_required",
                "possible_opposite_endpoint_forces_indeterminate",
                "direct_comparative_judgment_is_canonical_evidence",
                *_authority_data(),
                "spec_digest",
            },
            "shared-witness rubric spec",
        )
        if (
            raw["schema"] != SHARED_WITNESS_RUBRIC_SPEC_SCHEMA
            or raw["rubric_id"] != SHARED_WITNESS_RUBRIC_ID
            or raw["implementation_source_sha256"]
            != object_bongard_shared_witness_source_digest()
            or raw["observer_must_persist_witness_locator"] is not True
            or raw["observer_must_inventory_all_top_level_anchor_instances"]
            is not True
            or raw["observer_may_select_one_favorable_witness"] is not False
            or raw["observer_must_score_endpoints_separately"] is not True
            or raw["conservative_inventory_aggregation_required"] is not True
            or raw["possible_opposite_endpoint_forces_indeterminate"] is not True
            or raw["direct_comparative_judgment_is_canonical_evidence"] is not False
            or any(raw[key] != item for key, item in _authority_data().items())
        ):
            raise ObjectBongardSharedWitnessError(
                "shared-witness rubric spec policy differs"
            )
        result = cls(
            raw["semantic_artifact_digest"],
            raw["candidate_rank"],
            ObjectBongardSharedWitnessContrast.from_data(raw["contrast"]),
            raw["rubric"],
            raw["spec_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectBongardSharedWitnessError(
                "shared-witness rubric spec is not canonical"
            )
        return result


def build_shared_witness_rubric_specs(
    artifact: object,
    *,
    expected_artifact_digest: str,
) -> tuple[ObjectBongardSharedWitnessRubricSpec, ObjectBongardSharedWitnessRubricSpec]:
    """Strict bridge from the v1 structured artifact to two v2 specs."""

    from bongard.object_bongard_shared_witness_semantics import (
        ObjectBongardSharedWitnessSemanticArtifact,
    )
    from bongard.prototype_scene_observer import PrototypeSceneObserverStatus

    expected = _digest(expected_artifact_digest, "expected semantic artifact digest")
    if not isinstance(artifact, ObjectBongardSharedWitnessSemanticArtifact):
        raise TypeError("artifact must be a shared-witness semantic artifact")
    semantic = ObjectBongardSharedWitnessSemanticArtifact.from_data(
        artifact.to_data(), expected_artifact_digest=expected
    )
    if (
        semantic != artifact
        or semantic.status is not PrototypeSceneObserverStatus.SUCCESS
        or tuple(item.candidate_rank for item in semantic.contrast_candidates)
        != (0, 1)
    ):
        raise ObjectBongardSharedWitnessError(
            "shared-witness semantic artifact is not an accepted two-rank slate"
        )
    return tuple(
        ObjectBongardSharedWitnessRubricSpec.from_contrast(expected, contrast)
        for contrast in semantic.contrast_candidates
    )  # type: ignore[return-value]


__all__ = (
    "ObjectBongardSharedWitnessContrast",
    "ObjectBongardSharedWitnessError",
    "ObjectBongardSharedWitnessRubricSpec",
    "SHARED_WITNESS_CONTRAST_SCHEMA",
    "SHARED_WITNESS_IR_ID",
    "SHARED_WITNESS_RENDERER_ID",
    "SHARED_WITNESS_RUBRIC_ID",
    "SHARED_WITNESS_RUBRIC_SPEC_SCHEMA",
    "build_shared_witness_rubric_specs",
    "object_bongard_shared_witness_source_digest",
    "render_shared_witness_description",
    "render_shared_witness_rubric",
    "validate_axis_endpoint",
    "validate_shared_anchor",
    "validate_visual_axis",
)

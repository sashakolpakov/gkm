"""One-call, entity-grounded observer for structured shared-witness rubrics.

The vision call inventories every top-level coherent figure in one complete
panel and scores a shared anchor plus two neutrally presented endpoint cues on
each figure.  The model never sees which endpoint belongs to which Bongard
group.  After the receipted payload is frozen, deterministic Python maps the
neutral cue IDs back to the frozen rubric and projects the evidence to the
four dispositions.  Lean is not imported and has no role in the decision.
"""

from __future__ import annotations

from bongard.runtime_source_snapshot import capture_loaded_source, verify_loaded_source


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping

from bongard import prototype_scene_observer as _scene_runtime
from bongard.canonical import canonical_digest, canonical_json
from bongard.evidence import Disposition
from bongard.object_bongard_shared_witness import (
    ObjectBongardSharedWitnessRubricSpec,
    object_bongard_shared_witness_source_digest,
)
from bongard.prototype_scene_observer import (
    CloudPolicyCacheSnapshot,
    CodexModelCatalogSnapshot,
    CodexNoToolsAttestation,
    PrototypeImageIdentity,
    PrototypeSceneObserverStatus,
)
from bongard.python_predicate_authority import PYTHON_PREDICATE_AUTHORITY_ID
from bongard.transport import (
    CodexReceipt,
    NAMED_IMAGE_INPUT_DIGEST_SCHEMA,
    run_codex_named_images_structured,
    validate_codex_named_image_receipt,
    validate_codex_strict_output_schema,
)
from bongard.visual_witnesses import Q16BBox


SHARED_WITNESS_PANEL_ARTIFACT_SCHEMA = (
    "gkm.bongard-shared-witness-panel-observer-artifact.v1"
)
SHARED_WITNESS_PANEL_OBSERVATION_SCHEMA = (
    "gkm.bongard-shared-witness-panel-observation.v1"
)
SHARED_WITNESS_ENTITY_EVIDENCE_SCHEMA = (
    "gkm.bongard-shared-witness-entity-evidence.v1"
)
SHARED_WITNESS_PANEL_PROTOCOL_ID = (
    "bongard.shared-witness-panel/individual-entities-neutral-endpoints-v2"
)
SHARED_WITNESS_MAX_ENTITIES = 8

_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_ADDRESS = re.compile(r"sha256:[0-9a-f]{64}\Z")
_PANEL_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/-]{0,511}\Z")
_CODE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")
_ENTITY_ID = re.compile(r"e[0-7][0-9]\Z")
_CUE_ID = re.compile(r"cue_0[01]\Z")
_PROSE = re.compile(r"[ -~]+\Z")
_ROLE_WORD = re.compile(
    r"\b(?:group|target|foil|class|label|answer)\b",
    re.IGNORECASE,
)


class ObjectBongardSharedWitnessObserverError(ValueError):
    """A shared-witness panel observation or replay is malformed."""


def _authority_data() -> dict[str, object]:
    return {
        "predicate_authority_id": PYTHON_PREDICATE_AUTHORITY_ID,
        "python_is_canonical_authority": True,
        "model_inventories_all_top_level_entities": True,
        "anchor_scope": "individual_figure",
        "full_panel_scope_allowed": False,
        "model_can_self_authorize_anchor_scope": False,
        "overflow_requires_uncertain_inventory": True,
        "model_sees_neutral_content_sorted_endpoint_cues": True,
        "endpoint_group_mapping_model_visible": False,
        "projection_occurs_after_payload_freeze": True,
        "failed_fit_counts_as_absence": False,
        "negation_allowed": False,
        "polarity_flip_allowed": False,
        "threshold_selection_allowed": False,
        "lean_present": False,
        "lean_required": False,
        "lean_removable": True,
        "lean_defines_identity_or_decision": False,
        "lean_required_for_replay": False,
        "bbox_provenance": "model_generated_q16_no_verifier_pixel_witness_v1",
    }


def _fields(value: object, expected: set[str], label: str) -> Mapping[str, Any]:
    if (
        not isinstance(value, Mapping)
        or any(not isinstance(key, str) for key in value)
        or set(value) != expected
    ):
        raise ObjectBongardSharedWitnessObserverError(f"{label} fields differ")
    return value


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ObjectBongardSharedWitnessObserverError(
            f"{label} must be a raw lowercase SHA-256"
        )
    return value


def _address(value: object, label: str) -> str:
    if not isinstance(value, str) or _ADDRESS.fullmatch(value) is None:
        raise ObjectBongardSharedWitnessObserverError(
            f"{label} must be a sha256: address"
        )
    return value


def _panel_id(value: object) -> str:
    if not isinstance(value, str) or _PANEL_ID.fullmatch(value) is None:
        raise ObjectBongardSharedWitnessObserverError("panel ID is invalid")
    return value


def _bounded_prose(value: object, label: str, maximum: int) -> str:
    if (
        not isinstance(value, str)
        or not 3 <= len(value) <= maximum
        or value != value.strip()
        or "  " in value
        or _PROSE.fullmatch(value) is None
        or _ROLE_WORD.search(value) is not None
    ):
        raise ObjectBongardSharedWitnessObserverError(
            f"{label} violates bounded visible-prose grammar"
        )
    return value


def _canonical_payload(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(k, str) for k in value):
        raise ObjectBongardSharedWitnessObserverError("observer payload must be an object")
    try:
        decoded = json.loads(canonical_json(dict(value)).decode("utf-8"))
    except (TypeError, ValueError, UnicodeError, json.JSONDecodeError) as exc:
        raise ObjectBongardSharedWitnessObserverError(
            "observer payload is not canonical JSON"
        ) from exc
    if not isinstance(decoded, dict):
        raise ObjectBongardSharedWitnessObserverError("observer payload must be an object")
    return decoded


def _receipt_data(value: CodexReceipt | None) -> object:
    return None if value is None else value.to_dict()


def _receipt_from_data(value: object) -> CodexReceipt | None:
    if value is None:
        return None
    result = _scene_runtime._receipt_from_data(value)
    if not isinstance(result, CodexReceipt):
        raise ObjectBongardSharedWitnessObserverError("receipt has the wrong type")
    return result


def object_bongard_shared_witness_observer_source_digest() -> str:
    return verify_loaded_source(__name__, expected_source_sha256=_LOADED_SOURCE_SHA256)


def object_bongard_shared_witness_panel_observer_source_digest() -> str:
    """Command-facing explicit alias for the observer source identity."""

    return object_bongard_shared_witness_observer_source_digest()


@dataclass(frozen=True, slots=True)
class BinarySupportInterval:
    lower: int
    upper: int

    def __post_init__(self) -> None:
        if (
            type(self.lower) is not int
            or type(self.upper) is not int
            or self.lower not in (0, 1)
            or self.upper not in (0, 1)
            or self.lower > self.upper
        ):
            raise ObjectBongardSharedWitnessObserverError(
                "binary support interval differs"
            )

    @classmethod
    def from_judgment(cls, value: object) -> "BinarySupportInterval":
        try:
            return {
                "clear": cls(1, 1),
                "ambiguous": cls(0, 1),
                "none": cls(0, 0),
            }[value]  # type: ignore[index]
        except (KeyError, TypeError) as exc:
            raise ObjectBongardSharedWitnessObserverError(
                "support judgment is unknown"
            ) from exc

    def meet(self, other: "BinarySupportInterval") -> "BinarySupportInterval":
        if not isinstance(other, BinarySupportInterval):
            raise TypeError("support meet requires BinarySupportInterval")
        return BinarySupportInterval(min(self.lower, other.lower), min(self.upper, other.upper))

    def to_data(self) -> dict[str, int]:
        return {"lower": self.lower, "upper": self.upper}

    @classmethod
    def from_data(cls, value: object) -> "BinarySupportInterval":
        raw = _fields(value, {"lower", "upper"}, "binary support interval")
        return cls(raw["lower"], raw["upper"])


@dataclass(frozen=True, slots=True)
class ObjectBongardNeutralEndpointCue:
    cue_id: str
    text: str
    content_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.cue_id, str) or _CUE_ID.fullmatch(self.cue_id) is None:
            raise ObjectBongardSharedWitnessObserverError("neutral cue ID differs")
        _bounded_prose(self.text, "endpoint cue", 88)
        _digest(self.content_digest, "endpoint cue content digest")
        if self.content_digest != canonical_digest(
            {"schema": "gkm.bongard-neutral-endpoint-content.v1", "text": self.text}
        ):
            raise ObjectBongardSharedWitnessObserverError("endpoint cue digest differs")

    def to_data(self) -> dict[str, str]:
        return {
            "cue_id": self.cue_id,
            "text": self.text,
            "content_digest": self.content_digest,
        }

    @classmethod
    def from_data(cls, value: object) -> "ObjectBongardNeutralEndpointCue":
        raw = _fields(value, {"cue_id", "text", "content_digest"}, "neutral cue")
        return cls(raw["cue_id"], raw["text"], raw["content_digest"])


def _neutral_endpoint_cues(
    rubric_spec: ObjectBongardSharedWitnessRubricSpec,
) -> tuple[ObjectBongardNeutralEndpointCue, ObjectBongardNeutralEndpointCue]:
    if not isinstance(rubric_spec, ObjectBongardSharedWitnessRubricSpec):
        raise TypeError("rubric_spec must be ObjectBongardSharedWitnessRubricSpec")
    endpoints = (
        rubric_spec.contrast.group_0_endpoint,
        rubric_spec.contrast.group_1_endpoint,
    )
    rows = sorted(
        (
            canonical_digest(
                {"schema": "gkm.bongard-neutral-endpoint-content.v1", "text": text}
            ),
            text,
        )
        for text in endpoints
    )
    if rows[0][0] == rows[1][0]:
        raise ObjectBongardSharedWitnessObserverError("endpoint content digest collision")
    return tuple(
        ObjectBongardNeutralEndpointCue(f"cue_{index:02d}", text, digest)
        for index, (digest, text) in enumerate(rows)
    )  # type: ignore[return-value]


def _endpoint_mapping(
    rubric_spec: ObjectBongardSharedWitnessRubricSpec,
    cues: tuple[ObjectBongardNeutralEndpointCue, ObjectBongardNeutralEndpointCue],
) -> tuple[str, str]:
    by_text = {item.text: item.cue_id for item in cues}
    return (
        by_text[rubric_spec.contrast.group_0_endpoint],
        by_text[rubric_spec.contrast.group_1_endpoint],
    )


def object_bongard_shared_witness_panel_prompt(
    rubric_spec: ObjectBongardSharedWitnessRubricSpec,
) -> str:
    cues = _neutral_endpoint_cues(rubric_spec)
    rendered = "\n".join(f"- {item.cue_id}: {item.text}" for item in cues)
    contrast = rubric_spec.contrast
    return (
        "Inspect panel.png as one complete drawing. Inventory every top-level "
        "spatially coherent figure, including figures that do not match the named "
        "anchor. Use reading order, top-to-bottom then left-to-right, "
        "and assign consecutive IDs e00, e01, and so on with no gaps. Return at most "
        "the first eight figures; if more than eight exist, return those first eight "
        "and set inventory_status to uncertain. For each figure "
        "return its tight normalized Q16 bounding box, a short visible locator, an "
        "independent judgment of whether it is one individual "
        f"{contrast.shared_anchor}, and an independent judgment for each neutral cue "
        f"on that same individual's {contrast.visual_axis}. Judge clear, ambiguous, "
        "or none. Include short visible evidence for the anchor and for each cue. "
        "Never pool parts from separate figures. Never assign different cues to "
        "different figures as though they described one individual. "
        "Every entity has scope top_level_figure; never combine separate figures into "
        "one full-panel entity. Set inventory_status to uncertain if overlap, cropping, or "
        "visual ambiguity prevents a confident complete inventory. Endpoint meanings "
        "are intentionally neutral; do not infer hidden roles.\n\nNeutral endpoint cues, "
        "ordered by content digest:\n"
        f"{rendered}"
    )


def object_bongard_shared_witness_panel_output_schema(
    rubric_spec: ObjectBongardSharedWitnessRubricSpec,
) -> dict[str, object]:
    cue_ids = [item.cue_id for item in _neutral_endpoint_cues(rubric_spec)]
    bbox = {
        "type": "object",
        "properties": {
            name: {"type": "integer"} for name in ("x0", "y0", "x1", "y1")
        },
        "required": ["x0", "y0", "x1", "y1"],
        "additionalProperties": False,
    }
    cue_row = {
        "type": "object",
        "properties": {
            "cue_id": {"type": "string", "enum": cue_ids},
            "judgment": {
                "type": "string",
                "enum": ["clear", "ambiguous", "none"],
            },
            "evidence": {"type": "string"},
        },
        "required": ["cue_id", "judgment", "evidence"],
        "additionalProperties": False,
    }
    entity = {
        "type": "object",
        "properties": {
            "entity_id": {"type": "string"},
            "scope": {
                "type": "string",
                "enum": ["top_level_figure"],
            },
            "bbox_q16": bbox,
            "locator": {"type": "string"},
            "anchor_support": {
                "type": "string",
                "enum": ["clear", "ambiguous", "none"],
            },
            "anchor_evidence": {"type": "string"},
            "cue_support": {"type": "array", "items": cue_row},
        },
        "required": [
            "entity_id",
            "scope",
            "bbox_q16",
            "locator",
            "anchor_support",
            "anchor_evidence",
            "cue_support",
        ],
        "additionalProperties": False,
    }
    schema: dict[str, object] = {
        "type": "object",
        "properties": {
            "inventory_status": {
                "type": "string",
                "enum": ["complete", "uncertain"],
            },
            "entities": {"type": "array", "items": entity},
        },
        "required": ["inventory_status", "entities"],
        "additionalProperties": False,
    }
    validate_codex_strict_output_schema(schema)
    return schema


@dataclass(frozen=True, slots=True)
class ObjectBongardPreparedSharedWitnessPanelInputs:
    panel_digest: str
    rubric_spec_digest: str
    endpoint_cues: tuple[ObjectBongardNeutralEndpointCue, ObjectBongardNeutralEndpointCue]
    prompt: str
    output_schema: Mapping[str, Any]
    preparation_digest: str

    def __post_init__(self) -> None:
        _digest(self.panel_digest, "prepared panel digest")
        _digest(self.rubric_spec_digest, "prepared rubric spec digest")
        if tuple(sorted(self.endpoint_cues, key=lambda item: item.content_digest)) != self.endpoint_cues:
            raise ObjectBongardSharedWitnessObserverError("endpoint cues are not digest sorted")
        if tuple(item.cue_id for item in self.endpoint_cues) != ("cue_00", "cue_01"):
            raise ObjectBongardSharedWitnessObserverError("endpoint cue IDs differ")
        if not isinstance(self.prompt, str) or not isinstance(self.output_schema, Mapping):
            raise ObjectBongardSharedWitnessObserverError("prepared prompt/schema differs")
        _digest(self.preparation_digest, "preparation digest")
        expected = canonical_digest(
            {
                "schema": "gkm.bongard-prepared-shared-witness-panel-inputs.v1",
                "panel_digest": self.panel_digest,
                "rubric_spec_digest": self.rubric_spec_digest,
                "endpoint_cues": [item.to_data() for item in self.endpoint_cues],
                "prompt_digest": hashlib.sha256(self.prompt.encode("utf-8")).hexdigest(),
                "output_schema_digest": canonical_digest(dict(self.output_schema)),
            }
        )
        if self.preparation_digest != expected:
            raise ObjectBongardSharedWitnessObserverError("preparation digest differs")


def prepare_object_bongard_shared_witness_panel_inputs(
    png_bytes: bytes,
    rubric_spec: ObjectBongardSharedWitnessRubricSpec,
) -> ObjectBongardPreparedSharedWitnessPanelInputs:
    panel = _scene_runtime._validate_exact_png(png_bytes, "panel")
    if not isinstance(rubric_spec, ObjectBongardSharedWitnessRubricSpec):
        raise TypeError("rubric_spec must be ObjectBongardSharedWitnessRubricSpec")
    cues = _neutral_endpoint_cues(rubric_spec)
    prompt = object_bongard_shared_witness_panel_prompt(rubric_spec)
    schema = object_bongard_shared_witness_panel_output_schema(rubric_spec)
    values = {
        "panel_digest": hashlib.sha256(panel).hexdigest(),
        "rubric_spec_digest": rubric_spec.spec_digest,
        "endpoint_cues": cues,
        "prompt": prompt,
        "output_schema": schema,
    }
    provisional = object.__new__(ObjectBongardPreparedSharedWitnessPanelInputs)
    for name, item in values.items():
        object.__setattr__(provisional, name, item)
    digest = canonical_digest(
        {
            "schema": "gkm.bongard-prepared-shared-witness-panel-inputs.v1",
            "panel_digest": values["panel_digest"],
            "rubric_spec_digest": values["rubric_spec_digest"],
            "endpoint_cues": [item.to_data() for item in cues],
            "prompt_digest": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "output_schema_digest": canonical_digest(schema),
        }
    )
    return ObjectBongardPreparedSharedWitnessPanelInputs(**values, preparation_digest=digest)


@dataclass(frozen=True, slots=True)
class ObjectBongardSharedWitnessCueJudgment:
    cue_id: str
    judgment: str
    evidence: str
    interval: BinarySupportInterval

    def __post_init__(self) -> None:
        if not isinstance(self.cue_id, str) or _CUE_ID.fullmatch(self.cue_id) is None:
            raise ObjectBongardSharedWitnessObserverError("cue judgment ID differs")
        expected = BinarySupportInterval.from_judgment(self.judgment)
        _bounded_prose(self.evidence, "cue evidence", 180)
        if self.interval != expected:
            raise ObjectBongardSharedWitnessObserverError("cue interval differs")

    def to_data(self) -> dict[str, object]:
        return {
            "cue_id": self.cue_id,
            "judgment": self.judgment,
            "evidence": self.evidence,
            "interval": self.interval.to_data(),
        }

    @classmethod
    def from_data(cls, value: object) -> "ObjectBongardSharedWitnessCueJudgment":
        raw = _fields(
            value,
            {"cue_id", "judgment", "evidence", "interval"},
            "cue judgment",
        )
        return cls(
            raw["cue_id"],
            raw["judgment"],
            raw["evidence"],
            BinarySupportInterval.from_data(raw["interval"]),
        )


def _entity_content(value: "ObjectBongardSharedWitnessEntityEvidence") -> dict[str, object]:
    return {
        "schema": SHARED_WITNESS_ENTITY_EVIDENCE_SCHEMA,
        "entity_id": value.entity_id,
        "scope": value.scope,
        "bbox_q16": value.bbox_q16.to_data(),
        "locator": value.locator,
        "anchor_judgment": value.anchor_judgment,
        "anchor_evidence": value.anchor_evidence,
        "anchor_interval": value.anchor_interval.to_data(),
        "cue_judgments": [item.to_data() for item in value.cue_judgments],
        "target_interval": value.target_interval.to_data(),
        "foil_interval": value.foil_interval.to_data(),
        "pixel_witness_ids": list(value.pixel_witness_ids),
        "bbox_provenance": "model_generated_q16_no_verifier_pixel_witness_v1",
    }


@dataclass(frozen=True, slots=True)
class ObjectBongardSharedWitnessEntityEvidence:
    entity_id: str
    scope: str
    bbox_q16: Q16BBox
    locator: str
    anchor_judgment: str
    anchor_evidence: str
    anchor_interval: BinarySupportInterval
    cue_judgments: tuple[
        ObjectBongardSharedWitnessCueJudgment,
        ObjectBongardSharedWitnessCueJudgment,
    ]
    target_interval: BinarySupportInterval
    foil_interval: BinarySupportInterval
    pixel_witness_ids: tuple[str, ...]
    evidence_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.entity_id, str) or _ENTITY_ID.fullmatch(self.entity_id) is None:
            raise ObjectBongardSharedWitnessObserverError("entity ID differs")
        if self.scope != "top_level_figure":
            raise ObjectBongardSharedWitnessObserverError("entity scope differs")
        if not isinstance(self.bbox_q16, Q16BBox):
            raise TypeError("entity bbox must be Q16BBox")
        _bounded_prose(self.locator, "entity locator", 120)
        _bounded_prose(self.anchor_evidence, "anchor evidence", 180)
        if self.anchor_interval != BinarySupportInterval.from_judgment(
            self.anchor_judgment
        ):
            raise ObjectBongardSharedWitnessObserverError("anchor interval differs")
        if (
            not isinstance(self.cue_judgments, tuple)
            or len(self.cue_judgments) != 2
            or tuple(item.cue_id for item in self.cue_judgments)
            != ("cue_00", "cue_01")
        ):
            raise ObjectBongardSharedWitnessObserverError("entity cue judgments differ")
        combined = tuple(
            self.anchor_interval.meet(item.interval) for item in self.cue_judgments
        )
        if self.target_interval not in combined or self.foil_interval not in combined:
            raise ObjectBongardSharedWitnessObserverError(
                "target or foil interval is not an anchor-endpoint meet"
            )
        if self.pixel_witness_ids != ():
            raise ObjectBongardSharedWitnessObserverError(
                "v1 has no verifier-owned pixel witness attachment"
            )
        _digest(self.evidence_digest, "entity evidence digest")
        if self.evidence_digest != canonical_digest(_entity_content(self)):
            raise ObjectBongardSharedWitnessObserverError("entity evidence digest differs")

    @classmethod
    def create(
        cls,
        *,
        entity_id: str,
        scope: str,
        bbox_q16: Q16BBox,
        locator: str,
        anchor_judgment: str,
        anchor_evidence: str,
        cue_judgments: tuple[
            ObjectBongardSharedWitnessCueJudgment,
            ObjectBongardSharedWitnessCueJudgment,
        ],
        target_cue_id: str,
        foil_cue_id: str,
    ) -> "ObjectBongardSharedWitnessEntityEvidence":
        anchor = BinarySupportInterval.from_judgment(anchor_judgment)
        by_id = {item.cue_id: item.interval for item in cue_judgments}
        values = {
            "entity_id": entity_id,
            "scope": scope,
            "bbox_q16": bbox_q16,
            "locator": locator,
            "anchor_judgment": anchor_judgment,
            "anchor_evidence": anchor_evidence,
            "anchor_interval": anchor,
            "cue_judgments": cue_judgments,
            "target_interval": anchor.meet(by_id[target_cue_id]),
            "foil_interval": anchor.meet(by_id[foil_cue_id]),
            "pixel_witness_ids": (),
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(
            **values,
            evidence_digest=canonical_digest(_entity_content(provisional)),
        )

    def to_data(self) -> dict[str, object]:
        return {**_entity_content(self), "evidence_digest": self.evidence_digest}

    @classmethod
    def from_data(cls, value: object) -> "ObjectBongardSharedWitnessEntityEvidence":
        raw = _fields(
            value,
            {
                "schema",
                "entity_id",
                "scope",
                "bbox_q16",
                "locator",
                "anchor_judgment",
                "anchor_evidence",
                "anchor_interval",
                "cue_judgments",
                "target_interval",
                "foil_interval",
                "pixel_witness_ids",
                "bbox_provenance",
                "evidence_digest",
            },
            "entity evidence",
        )
        if (
            raw["schema"] != SHARED_WITNESS_ENTITY_EVIDENCE_SCHEMA
            or raw["bbox_provenance"]
            != "model_generated_q16_no_verifier_pixel_witness_v1"
            or not isinstance(raw["cue_judgments"], list)
            or not isinstance(raw["pixel_witness_ids"], list)
        ):
            raise ObjectBongardSharedWitnessObserverError("entity evidence policy differs")
        result = cls(
            raw["entity_id"],
            raw["scope"],
            Q16BBox.from_data(raw["bbox_q16"]),
            raw["locator"],
            raw["anchor_judgment"],
            raw["anchor_evidence"],
            BinarySupportInterval.from_data(raw["anchor_interval"]),
            tuple(
                ObjectBongardSharedWitnessCueJudgment.from_data(item)
                for item in raw["cue_judgments"]
            ),  # type: ignore[arg-type]
            BinarySupportInterval.from_data(raw["target_interval"]),
            BinarySupportInterval.from_data(raw["foil_interval"]),
            tuple(raw["pixel_witness_ids"]),
            raw["evidence_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectBongardSharedWitnessObserverError(
                "entity evidence is not canonical"
            )
        return result


def classify_shared_witness_entities(
    inventory_status: str,
    entities: tuple[ObjectBongardSharedWitnessEntityEvidence, ...],
) -> Disposition:
    """Closed, threshold-free Python projection over all inventoried entities."""

    if inventory_status not in ("complete", "uncertain"):
        raise ObjectBongardSharedWitnessObserverError("inventory status differs")
    if inventory_status == "uncertain":
        return Disposition.INDETERMINATE
    target_wins = bool(entities) and all(
        item.foil_interval.upper == 0 for item in entities
    ) and any(item.target_interval.lower == 1 for item in entities)
    foil_wins = bool(entities) and all(
        item.target_interval.upper == 0 for item in entities
    ) and any(item.foil_interval.lower == 1 for item in entities)
    if target_wins and not foil_wins:
        return Disposition.PRESENT
    if foil_wins and not target_wins:
        return Disposition.CERTIFIED_ABSENT
    return Disposition.INDETERMINATE


def _observation_content(
    value: "ObjectBongardSharedWitnessPanelObservation",
) -> dict[str, object]:
    return {
        "schema": SHARED_WITNESS_PANEL_OBSERVATION_SCHEMA,
        "rubric_spec_digest": value.rubric_spec_digest,
        "disposition": value.disposition.value,
        "inventory_status": value.inventory_status,
        "entities": [item.to_data() for item in value.entities],
        "target_cue_id": value.target_cue_id,
        "foil_cue_id": value.foil_cue_id,
        "payload_freeze_digest": value.payload_freeze_digest,
        "error_code": value.error_code,
        "error_type": value.error_type,
        "projection_id": "all-entities-anchor-endpoint-meet-v1",
    }


@dataclass(frozen=True, slots=True)
class ObjectBongardSharedWitnessPanelObservation:
    rubric_spec_digest: str
    disposition: Disposition
    inventory_status: str | None
    entities: tuple[ObjectBongardSharedWitnessEntityEvidence, ...]
    target_cue_id: str | None
    foil_cue_id: str | None
    payload_freeze_digest: str | None
    error_code: str | None
    error_type: str | None
    observation_digest: str

    def __post_init__(self) -> None:
        _digest(self.rubric_spec_digest, "observation rubric spec digest")
        if not isinstance(self.disposition, Disposition):
            raise TypeError("observation disposition has the wrong type")
        if self.disposition is Disposition.ERROR:
            if (
                self.inventory_status is not None
                or self.entities != ()
                or self.target_cue_id is not None
                or self.foil_cue_id is not None
                or self.payload_freeze_digest is not None
                or not isinstance(self.error_code, str)
                or _CODE.fullmatch(self.error_code) is None
                or not isinstance(self.error_type, str)
                or _CODE.fullmatch(self.error_type) is None
            ):
                raise ObjectBongardSharedWitnessObserverError("error observation differs")
        else:
            if (
                self.inventory_status not in ("complete", "uncertain")
                or not isinstance(self.entities, tuple)
                or len(self.entities) > SHARED_WITNESS_MAX_ENTITIES
                or tuple(item.entity_id for item in self.entities)
                != tuple(f"e{index:02d}" for index in range(len(self.entities)))
                or self.target_cue_id not in ("cue_00", "cue_01")
                or self.foil_cue_id not in ("cue_00", "cue_01")
                or self.target_cue_id == self.foil_cue_id
                or not isinstance(self.payload_freeze_digest, str)
                or self.error_code is not None
                or self.error_type is not None
            ):
                raise ObjectBongardSharedWitnessObserverError("scored observation differs")
            _digest(self.payload_freeze_digest, "payload freeze digest")
            for entity in self.entities:
                by_id = {item.cue_id: item.interval for item in entity.cue_judgments}
                if (
                    entity.target_interval
                    != entity.anchor_interval.meet(by_id[self.target_cue_id])
                    or entity.foil_interval
                    != entity.anchor_interval.meet(by_id[self.foil_cue_id])
                ):
                    raise ObjectBongardSharedWitnessObserverError(
                        "entity endpoint mapping differs"
                    )
            if classify_shared_witness_entities(self.inventory_status, self.entities) is not self.disposition:
                raise ObjectBongardSharedWitnessObserverError(
                    "observation disposition differs from entity evidence"
                )
        _digest(self.observation_digest, "observation digest")
        if self.observation_digest != canonical_digest(_observation_content(self)):
            raise ObjectBongardSharedWitnessObserverError("observation digest differs")

    @classmethod
    def error(
        cls, rubric_spec_digest: str, error_code: str, error_type: str
    ) -> "ObjectBongardSharedWitnessPanelObservation":
        values = {
            "rubric_spec_digest": rubric_spec_digest,
            "disposition": Disposition.ERROR,
            "inventory_status": None,
            "entities": (),
            "target_cue_id": None,
            "foil_cue_id": None,
            "payload_freeze_digest": None,
            "error_code": error_code,
            "error_type": error_type,
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(
            **values,
            observation_digest=canonical_digest(_observation_content(provisional)),
        )

    def to_data(self) -> dict[str, object]:
        return {**_observation_content(self), "observation_digest": self.observation_digest}

    @classmethod
    def from_data(cls, value: object) -> "ObjectBongardSharedWitnessPanelObservation":
        raw = _fields(
            value,
            {
                "schema",
                "rubric_spec_digest",
                "disposition",
                "inventory_status",
                "entities",
                "target_cue_id",
                "foil_cue_id",
                "payload_freeze_digest",
                "error_code",
                "error_type",
                "projection_id",
                "observation_digest",
            },
            "panel observation",
        )
        if (
            raw["schema"] != SHARED_WITNESS_PANEL_OBSERVATION_SCHEMA
            or raw["projection_id"] != "all-entities-anchor-endpoint-meet-v1"
            or not isinstance(raw["entities"], list)
        ):
            raise ObjectBongardSharedWitnessObserverError("observation policy differs")
        try:
            disposition = Disposition(raw["disposition"])
        except (TypeError, ValueError) as exc:
            raise ObjectBongardSharedWitnessObserverError(
                "observation disposition is unknown"
            ) from exc
        result = cls(
            raw["rubric_spec_digest"],
            disposition,
            raw["inventory_status"],
            tuple(
                ObjectBongardSharedWitnessEntityEvidence.from_data(item)
                for item in raw["entities"]
            ),
            raw["target_cue_id"],
            raw["foil_cue_id"],
            raw["payload_freeze_digest"],
            raw["error_code"],
            raw["error_type"],
            raw["observation_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectBongardSharedWitnessObserverError(
                "observation is not canonical"
            )
        return result


def _payload_freeze_digest(
    payload: Mapping[str, Any],
    receipt: CodexReceipt,
    endpoint_cues: tuple[ObjectBongardNeutralEndpointCue, ObjectBongardNeutralEndpointCue],
) -> str:
    return canonical_digest(
        {
            "schema": "gkm.bongard-shared-witness-panel-payload-freeze.v1",
            "model_payload": dict(payload),
            "receipt_digest": receipt.receipt_digest,
            "neutral_endpoint_cues": [item.to_data() for item in endpoint_cues],
            "endpoint_group_mapping_included": False,
        }
    )


def _project_frozen_payload(
    payload: Mapping[str, Any],
    *,
    rubric_spec: ObjectBongardSharedWitnessRubricSpec,
    endpoint_cues: tuple[
        ObjectBongardNeutralEndpointCue,
        ObjectBongardNeutralEndpointCue,
    ],
    payload_freeze_digest: str,
) -> ObjectBongardSharedWitnessPanelObservation:
    raw = _fields(payload, {"inventory_status", "entities"}, "observer payload")
    if raw["inventory_status"] not in ("complete", "uncertain") or not isinstance(
        raw["entities"], list
    ):
        raise ObjectBongardSharedWitnessObserverError("inventory payload differs")
    if len(raw["entities"]) > SHARED_WITNESS_MAX_ENTITIES:
        raise ObjectBongardSharedWitnessObserverError("entity inventory exceeds eight")
    target_cue_id, foil_cue_id = _endpoint_mapping(rubric_spec, endpoint_cues)
    entities: list[ObjectBongardSharedWitnessEntityEvidence] = []
    for index, value in enumerate(raw["entities"]):
        row = _fields(
            value,
            {
                "entity_id",
                "scope",
                "bbox_q16",
                "locator",
                "anchor_support",
                "anchor_evidence",
                "cue_support",
            },
            "entity payload",
        )
        expected_id = f"e{index:02d}"
        if row["entity_id"] != expected_id or not isinstance(row["cue_support"], list):
            raise ObjectBongardSharedWitnessObserverError(
                "entity IDs or cue inventory are not canonical"
            )
        judgments: list[ObjectBongardSharedWitnessCueJudgment] = []
        for cue_index, cue_value in enumerate(row["cue_support"]):
            cue_row = _fields(
                cue_value,
                {"cue_id", "judgment", "evidence"},
                "cue support payload",
            )
            expected_cue_id = f"cue_{cue_index:02d}"
            if cue_row["cue_id"] != expected_cue_id:
                raise ObjectBongardSharedWitnessObserverError(
                    "cue support order differs from neutral presentation"
                )
            judgments.append(
                ObjectBongardSharedWitnessCueJudgment(
                    cue_row["cue_id"],
                    cue_row["judgment"],
                    cue_row["evidence"],
                    BinarySupportInterval.from_judgment(cue_row["judgment"]),
                )
            )
        if len(judgments) != 2:
            raise ObjectBongardSharedWitnessObserverError(
                "each entity must judge both endpoint cues"
            )
        entities.append(
            ObjectBongardSharedWitnessEntityEvidence.create(
                entity_id=row["entity_id"],
                scope=row["scope"],
                bbox_q16=Q16BBox.from_data(row["bbox_q16"]),
                locator=row["locator"],
                anchor_judgment=row["anchor_support"],
                anchor_evidence=row["anchor_evidence"],
                cue_judgments=tuple(judgments),  # type: ignore[arg-type]
                target_cue_id=target_cue_id,
                foil_cue_id=foil_cue_id,
            )
        )
    frozen_entities = tuple(entities)
    values = {
        "rubric_spec_digest": rubric_spec.spec_digest,
        "disposition": classify_shared_witness_entities(
            raw["inventory_status"], frozen_entities
        ),
        "inventory_status": raw["inventory_status"],
        "entities": frozen_entities,
        "target_cue_id": target_cue_id,
        "foil_cue_id": foil_cue_id,
        "payload_freeze_digest": _digest(
            payload_freeze_digest, "payload freeze digest"
        ),
        "error_code": None,
        "error_type": None,
    }
    provisional = object.__new__(ObjectBongardSharedWitnessPanelObservation)
    for name, item in values.items():
        object.__setattr__(provisional, name, item)
    return ObjectBongardSharedWitnessPanelObservation(
        **values,
        observation_digest=canonical_digest(_observation_content(provisional)),
    )


def object_bongard_shared_witness_panel_protocol_digest() -> str:
    """Identify the complete one-call observation and Python projection policy."""

    return canonical_digest(
        {
            "schema": "gkm.bongard-shared-witness-panel-protocol.v1",
            "protocol_id": SHARED_WITNESS_PANEL_PROTOCOL_ID,
            "observer_source_digest": object_bongard_shared_witness_observer_source_digest(),
            "shared_witness_source_digest": object_bongard_shared_witness_source_digest(),
            "physical_calls_per_panel": 1,
            "whole_panel_only": True,
            "maximum_entities": SHARED_WITNESS_MAX_ENTITIES,
            "anchor_scope": "individual_figure",
            "full_panel_scope_allowed": False,
            "overflow_policy": "first-eight-and-uncertain",
            "entity_ids": [f"e{index:02d}" for index in range(SHARED_WITNESS_MAX_ENTITIES)],
            "support_map": {
                "clear": [1, 1],
                "ambiguous": [0, 1],
                "none": [0, 0],
            },
            "entity_endpoint_support": "meet(anchor,endpoint)",
            "present_rule": (
                "some-target-lower-one-and-every-foil-upper-zero"
            ),
            "certified_absent_rule": (
                "some-foil-lower-one-and-every-target-upper-zero"
            ),
            "uncertain_inventory_is_indeterminate": True,
            "empty_inventory_is_indeterminate": True,
            "model_output_contains_signed_panel_decision": False,
            "output_schema_is_spec_instantiated": True,
            **_authority_data(),
        }
    )


def _model_digest(model: str, reasoning_effort: str) -> str:
    if not isinstance(model, str) or not model or not isinstance(reasoning_effort, str) or not reasoning_effort:
        raise ObjectBongardSharedWitnessObserverError("model request differs")
    return canonical_digest(
        {
            "schema": "gkm.bongard-shared-witness-panel-model.v1",
            "model": model,
            "reasoning_effort": reasoning_effort,
        }
    )


def _runtime_identity_digest(
    *,
    model: str,
    reasoning_effort: str,
    expected_launcher_digest: str | None,
    cloud_policy_cache_binding: str,
    model_catalog_digest: str,
    no_tools_attestation_digest: str,
) -> str:
    return canonical_digest(
        {
            "schema": "gkm.bongard-shared-witness-panel-runtime.v1",
            "model_digest": _model_digest(model, reasoning_effort),
            "expected_launcher_digest": expected_launcher_digest,
            "cloud_policy_cache_binding": cloud_policy_cache_binding,
            "model_catalog_digest": model_catalog_digest,
            "no_tools_attestation_digest": no_tools_attestation_digest,
            "transport_source_digest": _scene_runtime.prototype_scene_transport_source_digest(),
        }
    )


def _artifact_content(value: "ObjectBongardSharedWitnessPanelArtifact") -> dict[str, object]:
    return {
        "schema": SHARED_WITNESS_PANEL_ARTIFACT_SCHEMA,
        "panel_id": value.panel_id,
        "panel_digest": value.panel_digest,
        "observation_context_digest": value.observation_context_digest,
        "rubric_spec": value.rubric_spec.to_data(),
        "rubric_spec_digest": value.rubric_spec_digest,
        "endpoint_cues": [item.to_data() for item in value.endpoint_cues],
        "preparation_digest": value.preparation_digest,
        "source_digest": value.source_digest,
        "shared_witness_source_digest": value.shared_witness_source_digest,
        "protocol_digest": value.protocol_digest,
        "transport_source_digest": value.transport_source_digest,
        "prompt_digest": value.prompt_digest,
        "output_schema_digest": value.output_schema_digest,
        "model": value.model,
        "reasoning_effort": value.reasoning_effort,
        "model_digest": value.model_digest,
        "expected_launcher_digest": value.expected_launcher_digest,
        "cloud_policy_cache_binding": value.cloud_policy_cache_binding,
        "model_catalog_digest": value.model_catalog_digest,
        "no_tools_attestation_digest": value.no_tools_attestation_digest,
        "runtime_identity_digest": value.runtime_identity_digest,
        "presentation": [item.to_data() for item in value.presentation],
        "physical_call_count": value.physical_call_count,
        "status": value.status.value,
        "model_payload": value.model_payload,
        "receipt": _receipt_data(value.receipt),
        "observation": value.observation.to_data(),
        "failure_code": value.failure_code,
        "failure_type": value.failure_type,
        "whole_panel_only": True,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectBongardSharedWitnessPanelArtifact:
    panel_id: str
    panel_digest: str
    observation_context_digest: str
    rubric_spec: ObjectBongardSharedWitnessRubricSpec
    rubric_spec_digest: str
    endpoint_cues: tuple[ObjectBongardNeutralEndpointCue, ObjectBongardNeutralEndpointCue]
    preparation_digest: str
    source_digest: str
    shared_witness_source_digest: str
    protocol_digest: str
    transport_source_digest: str
    prompt_digest: str
    output_schema_digest: str
    model: str
    reasoning_effort: str
    model_digest: str
    expected_launcher_digest: str | None
    cloud_policy_cache_binding: str
    model_catalog_digest: str
    no_tools_attestation_digest: str
    runtime_identity_digest: str
    presentation: tuple[PrototypeImageIdentity, ...]
    physical_call_count: int
    status: PrototypeSceneObserverStatus
    model_payload: Mapping[str, Any] | None
    receipt: CodexReceipt | None
    observation: ObjectBongardSharedWitnessPanelObservation
    failure_code: str | None
    failure_type: str | None
    artifact_digest: str

    def __post_init__(self) -> None:
        _panel_id(self.panel_id)
        _digest(self.panel_digest, "panel digest")
        _address(self.observation_context_digest, "observation context")
        if not isinstance(self.rubric_spec, ObjectBongardSharedWitnessRubricSpec):
            raise TypeError("rubric spec has the wrong type")
        for name in (
            "rubric_spec_digest", "preparation_digest", "source_digest",
            "shared_witness_source_digest", "protocol_digest", "transport_source_digest",
            "prompt_digest", "output_schema_digest", "model_digest", "model_catalog_digest",
            "no_tools_attestation_digest", "runtime_identity_digest", "artifact_digest",
        ):
            _digest(getattr(self, name), name)
        if self.expected_launcher_digest is None:
            raise ObjectBongardSharedWitnessObserverError("launcher commitment is required")
        _digest(self.expected_launcher_digest, "launcher digest")
        if self.cloud_policy_cache_binding != "absent":
            _address(self.cloud_policy_cache_binding, "cloud policy binding")
        expected_cues = _neutral_endpoint_cues(self.rubric_spec)
        target_cue_id, foil_cue_id = _endpoint_mapping(self.rubric_spec, expected_cues)
        prompt = object_bongard_shared_witness_panel_prompt(self.rubric_spec)
        schema = object_bongard_shared_witness_panel_output_schema(self.rubric_spec)
        expected_preparation = canonical_digest(
            {
                "schema": "gkm.bongard-prepared-shared-witness-panel-inputs.v1",
                "panel_digest": self.panel_digest,
                "rubric_spec_digest": self.rubric_spec.spec_digest,
                "endpoint_cues": [item.to_data() for item in expected_cues],
                "prompt_digest": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "output_schema_digest": canonical_digest(schema),
            }
        )
        expected_runtime = _runtime_identity_digest(
            model=self.model,
            reasoning_effort=self.reasoning_effort,
            expected_launcher_digest=self.expected_launcher_digest,
            cloud_policy_cache_binding=self.cloud_policy_cache_binding,
            model_catalog_digest=self.model_catalog_digest,
            no_tools_attestation_digest=self.no_tools_attestation_digest,
        )
        if (
            self.rubric_spec_digest != self.rubric_spec.spec_digest
            or self.endpoint_cues != expected_cues
            or self.preparation_digest != expected_preparation
            or self.source_digest != object_bongard_shared_witness_observer_source_digest()
            or self.shared_witness_source_digest != object_bongard_shared_witness_source_digest()
            or self.protocol_digest != object_bongard_shared_witness_panel_protocol_digest()
            or self.transport_source_digest != _scene_runtime.prototype_scene_transport_source_digest()
            or self.prompt_digest != hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            or self.output_schema_digest != canonical_digest(schema)
            or self.model_digest != _model_digest(self.model, self.reasoning_effort)
            or self.runtime_identity_digest != expected_runtime
        ):
            raise ObjectBongardSharedWitnessObserverError("artifact protocol binding differs")
        if (
            not isinstance(self.presentation, tuple) or len(self.presentation) != 1
            or self.presentation[0].name != "panel.png"
            or self.presentation[0].content_digest != self.panel_digest
            or self.physical_call_count != 1
            or not isinstance(self.status, PrototypeSceneObserverStatus)
            or not isinstance(self.observation, ObjectBongardSharedWitnessPanelObservation)
            or self.observation.rubric_spec_digest != self.rubric_spec_digest
        ):
            raise ObjectBongardSharedWitnessObserverError("artifact observation binding differs")
        if self.model_payload is not None:
            object.__setattr__(self, "model_payload", _canonical_payload(self.model_payload))
        if self.status is PrototypeSceneObserverStatus.SUCCESS:
            expected_freeze_digest = (
                None
                if self.model_payload is None or self.receipt is None
                else _payload_freeze_digest(
                    self.model_payload,
                    self.receipt,
                    expected_cues,
                )
            )
            expected_observation = (
                None
                if expected_freeze_digest is None
                else _project_frozen_payload(
                    self.model_payload,
                    rubric_spec=self.rubric_spec,
                    endpoint_cues=expected_cues,
                    payload_freeze_digest=expected_freeze_digest,
                )
            )
            if (
                self.model_payload is None or not isinstance(self.receipt, CodexReceipt)
                or self.failure_code is not None or self.failure_type is not None
                or self.observation.target_cue_id != target_cue_id
                or self.observation.foil_cue_id != foil_cue_id
                or self.observation != expected_observation
            ):
                raise ObjectBongardSharedWitnessObserverError("successful artifact differs")
        elif self.status in {PrototypeSceneObserverStatus.PARSER_ERROR, PrototypeSceneObserverStatus.TRANSPORT_ERROR}:
            if (
                self.observation.disposition is not Disposition.ERROR
                or self.observation.target_cue_id is not None
                or self.observation.foil_cue_id is not None
                or self.observation.error_code != self.failure_code
                or self.observation.error_type != self.failure_type
                or not isinstance(self.failure_code, str) or _CODE.fullmatch(self.failure_code) is None
                or not isinstance(self.failure_type, str) or _CODE.fullmatch(self.failure_type) is None
            ):
                raise ObjectBongardSharedWitnessObserverError("failed artifact lacks typed error")
            if self.status is PrototypeSceneObserverStatus.PARSER_ERROR and (self.model_payload is None or self.receipt is None):
                raise ObjectBongardSharedWitnessObserverError("parser failure lacks payload receipt")
            if self.status is PrototypeSceneObserverStatus.TRANSPORT_ERROR and (self.model_payload is not None or self.receipt is not None):
                raise ObjectBongardSharedWitnessObserverError("transport failure contains payload")
            if self.status is PrototypeSceneObserverStatus.PARSER_ERROR:
                if (
                    self.failure_code != "observer_payload_rejected"
                    or not isinstance(self.receipt, CodexReceipt)
                ):
                    raise ObjectBongardSharedWitnessObserverError(
                        "parser failure identity differs"
                    )
                assert self.model_payload is not None
                try:
                    failed_freeze_digest = _payload_freeze_digest(
                        self.model_payload,
                        self.receipt,
                        expected_cues,
                    )
                    _project_frozen_payload(
                        self.model_payload,
                        rubric_spec=self.rubric_spec,
                        endpoint_cues=expected_cues,
                        payload_freeze_digest=failed_freeze_digest,
                    )
                except Exception as exc:
                    if self.failure_type != _scene_runtime._exception_type(exc):
                        raise ObjectBongardSharedWitnessObserverError(
                            "parser failure type differs from deterministic replay"
                        ) from exc
                else:
                    raise ObjectBongardSharedWitnessObserverError(
                        "parser failure payload projects successfully"
                    )
            elif self.failure_code != "observer_transport_failed":
                raise ObjectBongardSharedWitnessObserverError(
                    "transport failure identity differs"
                )
        else:
            raise ObjectBongardSharedWitnessObserverError("artifact status differs")
        if self.receipt is not None:
            view = [item.to_data() for item in self.presentation]
            expected_set = "sha256:" + canonical_digest(
                {"schema": NAMED_IMAGE_INPUT_DIGEST_SCHEMA, "images": view}
            )
            if (
                self.receipt.prompt_digest != self.prompt_digest
                or self.receipt.output_schema_digest != self.output_schema_digest
                or self.receipt.structured_output_digest != canonical_digest(dict(self.model_payload or {}))
                or self.receipt.panel_view_digest != canonical_digest(view)
                or self.receipt.panel_set_digest != expected_set
                or self.receipt.requested_model != self.model
                or self.receipt.requested_reasoning_effort != self.reasoning_effort
                or self.receipt.codex_launcher_digest != self.expected_launcher_digest
                or self.receipt.cloud_config_bundle_cache_binding != self.cloud_policy_cache_binding
                or self.receipt.model_catalog_digest != self.model_catalog_digest
                or self.receipt.tool_surface_attestation_digest != self.no_tools_attestation_digest
            ):
                raise ObjectBongardSharedWitnessObserverError("artifact receipt binding differs")
        if self.artifact_digest != canonical_digest(_artifact_content(self)):
            raise ObjectBongardSharedWitnessObserverError("artifact digest differs")

    def to_data(self) -> dict[str, object]:
        return {**_artifact_content(self), "artifact_digest": self.artifact_digest}

    @classmethod
    def from_data(cls, value: object) -> "ObjectBongardSharedWitnessPanelArtifact":
        fields = {
            "schema", "panel_id", "panel_digest", "observation_context_digest", "rubric_spec",
            "rubric_spec_digest", "endpoint_cues", "preparation_digest", "source_digest",
            "shared_witness_source_digest", "protocol_digest", "transport_source_digest",
            "prompt_digest", "output_schema_digest", "model", "reasoning_effort", "model_digest",
            "expected_launcher_digest", "cloud_policy_cache_binding", "model_catalog_digest",
            "no_tools_attestation_digest", "runtime_identity_digest", "presentation",
            "physical_call_count", "status", "model_payload", "receipt", "observation",
            "failure_code", "failure_type", "whole_panel_only", *_authority_data(), "artifact_digest",
        }
        raw = _fields(value, fields, "panel artifact")
        if (
            raw["schema"] != SHARED_WITNESS_PANEL_ARTIFACT_SCHEMA
            or raw["whole_panel_only"] is not True
            or any(raw[key] != item for key, item in _authority_data().items())
            or not isinstance(raw["endpoint_cues"], list)
            or not isinstance(raw["presentation"], list)
        ):
            raise ObjectBongardSharedWitnessObserverError("panel artifact policy differs")
        try:
            status = PrototypeSceneObserverStatus(raw["status"])
        except (TypeError, ValueError) as exc:
            raise ObjectBongardSharedWitnessObserverError("unknown artifact status") from exc
        result = cls(
            raw["panel_id"], raw["panel_digest"], raw["observation_context_digest"],
            ObjectBongardSharedWitnessRubricSpec.from_data(raw["rubric_spec"]), raw["rubric_spec_digest"],
            tuple(ObjectBongardNeutralEndpointCue.from_data(item) for item in raw["endpoint_cues"]),
            raw["preparation_digest"], raw["source_digest"], raw["shared_witness_source_digest"],
            raw["protocol_digest"], raw["transport_source_digest"], raw["prompt_digest"],
            raw["output_schema_digest"], raw["model"], raw["reasoning_effort"], raw["model_digest"],
            raw["expected_launcher_digest"], raw["cloud_policy_cache_binding"], raw["model_catalog_digest"],
            raw["no_tools_attestation_digest"], raw["runtime_identity_digest"],
            tuple(PrototypeImageIdentity.from_data(item) for item in raw["presentation"]),
            raw["physical_call_count"], status, raw["model_payload"], _receipt_from_data(raw["receipt"]),
            ObjectBongardSharedWitnessPanelObservation.from_data(raw["observation"]),
            raw["failure_code"], raw["failure_type"], raw["artifact_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectBongardSharedWitnessObserverError("panel artifact is not canonical")
        return result


def _seal_artifact(
    *,
    panel_id: str,
    prepared: ObjectBongardPreparedSharedWitnessPanelInputs,
    context: str,
    rubric_spec: ObjectBongardSharedWitnessRubricSpec,
    model: str,
    reasoning_effort: str,
    expected_launcher_digest: str | None,
    cloud_policy_cache_binding: str,
    model_catalog_digest: str,
    no_tools_attestation_digest: str,
    presentation: tuple[PrototypeImageIdentity, ...],
    status: PrototypeSceneObserverStatus,
    payload: Mapping[str, Any] | None,
    receipt: CodexReceipt | None,
    observation: ObjectBongardSharedWitnessPanelObservation,
    failure_code: str | None,
    failure_type: str | None,
) -> ObjectBongardSharedWitnessPanelArtifact:
    values = {
        "panel_id": panel_id,
        "panel_digest": prepared.panel_digest,
        "observation_context_digest": context,
        "rubric_spec": rubric_spec,
        "rubric_spec_digest": rubric_spec.spec_digest,
        "endpoint_cues": prepared.endpoint_cues,
        "preparation_digest": prepared.preparation_digest,
        "source_digest": object_bongard_shared_witness_observer_source_digest(),
        "shared_witness_source_digest": object_bongard_shared_witness_source_digest(),
        "protocol_digest": object_bongard_shared_witness_panel_protocol_digest(),
        "transport_source_digest": _scene_runtime.prototype_scene_transport_source_digest(),
        "prompt_digest": hashlib.sha256(prepared.prompt.encode("utf-8")).hexdigest(),
        "output_schema_digest": canonical_digest(dict(prepared.output_schema)),
        "model": model,
        "reasoning_effort": reasoning_effort,
        "model_digest": _model_digest(model, reasoning_effort),
        "expected_launcher_digest": expected_launcher_digest,
        "cloud_policy_cache_binding": cloud_policy_cache_binding,
        "model_catalog_digest": model_catalog_digest,
        "no_tools_attestation_digest": no_tools_attestation_digest,
        "runtime_identity_digest": _runtime_identity_digest(
            model=model, reasoning_effort=reasoning_effort,
            expected_launcher_digest=expected_launcher_digest,
            cloud_policy_cache_binding=cloud_policy_cache_binding,
            model_catalog_digest=model_catalog_digest,
            no_tools_attestation_digest=no_tools_attestation_digest,
        ),
        "presentation": presentation,
        "physical_call_count": 1,
        "status": status,
        "model_payload": None if payload is None else _canonical_payload(payload),
        "receipt": receipt,
        "observation": observation,
        "failure_code": failure_code,
        "failure_type": failure_type,
    }
    provisional = object.__new__(ObjectBongardSharedWitnessPanelArtifact)
    for name, item in values.items():
        object.__setattr__(provisional, name, item)
    return ObjectBongardSharedWitnessPanelArtifact(
        **values, artifact_digest=canonical_digest(_artifact_content(provisional))
    )


def observe_object_bongard_shared_witness_panel(
    png_bytes: bytes,
    *,
    panel_id: str,
    rubric_spec: ObjectBongardSharedWitnessRubricSpec,
    expected_panel_sha256: str,
    expected_rubric_spec_digest: str,
    model: str,
    reasoning_effort: str = "medium",
    minutes: int = 15,
    verbose: bool = False,
    executable: str = "codex",
    cloud_policy_cache_snapshot: CloudPolicyCacheSnapshot | None = None,
    expected_launcher_digest: str | None = None,
    model_catalog_snapshot: CodexModelCatalogSnapshot,
    no_tools_attestation: CodexNoToolsAttestation,
    transport=run_codex_named_images_structured,
    observation_context_digest: str | None = None,
) -> ObjectBongardSharedWitnessPanelArtifact:
    """Inventory and independently judge every entity in one complete panel."""

    panel = _scene_runtime._validate_exact_png(png_bytes, "panel")
    identity = _panel_id(panel_id)
    prepared = prepare_object_bongard_shared_witness_panel_inputs(panel, rubric_spec)
    if prepared.panel_digest != _digest(expected_panel_sha256, "expected panel digest"):
        raise ObjectBongardSharedWitnessObserverError("panel differs from commitment")
    if rubric_spec.spec_digest != _digest(expected_rubric_spec_digest, "expected rubric digest"):
        raise ObjectBongardSharedWitnessObserverError("rubric differs from commitment")
    if not callable(transport):
        raise TypeError("transport must be callable")
    policy = _scene_runtime._policy_cache_binding(cloud_policy_cache_snapshot)
    model_catalog_digest, no_tools_digest = _scene_runtime._validate_no_tools_runtime(
        model_catalog_snapshot=model_catalog_snapshot,
        no_tools_attestation=no_tools_attestation,
        expected_launcher_digest=expected_launcher_digest,
        cloud_policy_cache_binding=policy,
    )
    context = observation_context_digest or "sha256:" + canonical_digest(
        {
            "schema": "gkm.bongard-shared-witness-panel-context.v1",
            "panel_id": identity,
            "panel_digest": prepared.panel_digest,
            "rubric_spec_digest": rubric_spec.spec_digest,
            "model": model,
            "reasoning_effort": reasoning_effort,
        }
    )
    _address(context, "observation context")
    presentation_bytes = (("panel.png", panel),)
    presentation = _scene_runtime._image_identities(presentation_bytes)
    _scene_runtime._assert_model_visible_boundary(
        prepared.prompt,
        dict(prepared.output_schema),
        ("panel.png",),
        hidden_values=(identity, prepared.panel_digest, rubric_spec.spec_digest, context),
    )
    common = {
        "panel_id": identity, "prepared": prepared, "context": context,
        "rubric_spec": rubric_spec, "model": model, "reasoning_effort": reasoning_effort,
        "expected_launcher_digest": expected_launcher_digest,
        "cloud_policy_cache_binding": policy, "model_catalog_digest": model_catalog_digest,
        "no_tools_attestation_digest": no_tools_digest, "presentation": presentation,
    }
    try:
        payload, receipt = _scene_runtime._stage_and_call(
            presentation_bytes,
            prompt=prepared.prompt,
            schema=dict(prepared.output_schema),
            model=model,
            reasoning_effort=reasoning_effort,
            minutes=minutes,
            verbose=verbose,
            executable=executable,
            cloud_policy_cache_snapshot=cloud_policy_cache_snapshot,
            expected_launcher_digest=expected_launcher_digest,
            model_catalog_snapshot=model_catalog_snapshot,
            no_tools_attestation=no_tools_attestation,
            transport=transport,
        )
    except Exception as exc:
        error_type = _scene_runtime._exception_type(exc)
        return _seal_artifact(
            **common,
            status=PrototypeSceneObserverStatus.TRANSPORT_ERROR,
            payload=None,
            receipt=None,
            observation=ObjectBongardSharedWitnessPanelObservation.error(
                rubric_spec.spec_digest,
                "observer_transport_failed",
                error_type,
            ),
            failure_code="observer_transport_failed",
            failure_type=error_type,
        )
    try:
        payload_freeze_digest = _payload_freeze_digest(
            payload,
            receipt,
            prepared.endpoint_cues,
        )
        observation = _project_frozen_payload(
            payload,
            rubric_spec=rubric_spec,
            endpoint_cues=prepared.endpoint_cues,
            payload_freeze_digest=payload_freeze_digest,
        )
    except Exception as exc:
        error_type = _scene_runtime._exception_type(exc)
        return _seal_artifact(
            **common,
            status=PrototypeSceneObserverStatus.PARSER_ERROR,
            payload=payload,
            receipt=receipt,
            observation=ObjectBongardSharedWitnessPanelObservation.error(
                rubric_spec.spec_digest,
                "observer_payload_rejected",
                error_type,
            ),
            failure_code="observer_payload_rejected",
            failure_type=error_type,
        )
    return _seal_artifact(
        **common,
        status=PrototypeSceneObserverStatus.SUCCESS,
        payload=payload,
        receipt=receipt,
        observation=observation,
        failure_code=None,
        failure_type=None,
    )


def verify_object_bongard_shared_witness_panel_artifact(
    artifact: ObjectBongardSharedWitnessPanelArtifact,
    png_bytes: bytes,
    *,
    panel_id: str,
    rubric_spec: ObjectBongardSharedWitnessRubricSpec,
    expected_artifact_digest: str,
    expected_runtime_identity_digest: str | None = None,
) -> ObjectBongardSharedWitnessPanelArtifact:
    """Cold-replay pixels, neutralization, parsing, projection, and receipt."""

    if not isinstance(artifact, ObjectBongardSharedWitnessPanelArtifact):
        raise TypeError("artifact must be ObjectBongardSharedWitnessPanelArtifact")
    restored = ObjectBongardSharedWitnessPanelArtifact.from_data(artifact.to_data())
    if restored.artifact_digest != _digest(expected_artifact_digest, "expected artifact digest"):
        raise ObjectBongardSharedWitnessObserverError("artifact differs from commitment")
    if expected_runtime_identity_digest is not None and restored.runtime_identity_digest != _digest(
        expected_runtime_identity_digest, "expected runtime digest"
    ):
        raise ObjectBongardSharedWitnessObserverError("runtime differs from commitment")
    panel = _scene_runtime._validate_exact_png(png_bytes, "panel")
    spec = ObjectBongardSharedWitnessRubricSpec.from_data(rubric_spec.to_data())
    if (
        restored.panel_id != _panel_id(panel_id)
        or restored.panel_digest != hashlib.sha256(panel).hexdigest()
        or restored.presentation[0].byte_count != len(panel)
        or restored.rubric_spec != spec
    ):
        raise ObjectBongardSharedWitnessObserverError("cold replay inputs differ")
    prepared = prepare_object_bongard_shared_witness_panel_inputs(panel, spec)
    if prepared.preparation_digest != restored.preparation_digest:
        raise ObjectBongardSharedWitnessObserverError("cold preparation differs")
    if restored.receipt is not None:
        assert restored.model_payload is not None
        with tempfile.TemporaryDirectory(prefix="bongard-shared-witness-replay-") as raw:
            target = Path(raw) / "panel.png"
            target.write_bytes(panel)
            validate_codex_named_image_receipt(
                restored.receipt,
                prepared.prompt,
                (str(target.resolve()),),
                ("panel.png",),
                dict(prepared.output_schema),
                dict(restored.model_payload),
            )
            if target.read_bytes() != panel:
                raise ObjectBongardSharedWitnessObserverError("cold replay panel changed")
    return restored


# Descriptive aliases used by the calibration/campaign orchestration layers.
object_bongard_shared_witness_panel_observer_source_digest = (
    object_bongard_shared_witness_observer_source_digest
)


__all__ = (
    "BinarySupportInterval",
    "ObjectBongardNeutralEndpointCue",
    "ObjectBongardPreparedSharedWitnessPanelInputs",
    "ObjectBongardSharedWitnessCueJudgment",
    "ObjectBongardSharedWitnessEntityEvidence",
    "ObjectBongardSharedWitnessObserverError",
    "ObjectBongardSharedWitnessPanelArtifact",
    "ObjectBongardSharedWitnessPanelObservation",
    "SHARED_WITNESS_MAX_ENTITIES",
    "SHARED_WITNESS_PANEL_PROTOCOL_ID",
    "classify_shared_witness_entities",
    "object_bongard_shared_witness_observer_source_digest",
    "object_bongard_shared_witness_panel_observer_source_digest",
    "object_bongard_shared_witness_panel_output_schema",
    "object_bongard_shared_witness_panel_prompt",
    "object_bongard_shared_witness_panel_protocol_digest",
    "observe_object_bongard_shared_witness_panel",
    "prepare_object_bongard_shared_witness_panel_inputs",
    "verify_object_bongard_shared_witness_panel_artifact",
)

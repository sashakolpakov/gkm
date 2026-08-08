"""Predicate-independent object proposals and one-call visual transcripts.

The deterministic half freezes exact-pixel proposal lineages and a crop atlas
before any visual description exists.  Proposal lineages are deliberately not
called semantic objects: eligible singleton and persistent-union lineages can
overlap, and that overlap is retained as an explicit graph.  The empirical
half makes exactly one neutral no-tools vision call over the complete panel and
the frozen crop atlas.  It returns a whole-panel summary and bounded tags plus
qualitative, interval-valued observations and bounded tags for every proposal.

Python owns extraction, parsing, projection, identity, and replay.  Lean is
not imported and is neither required nor consulted.
"""

from __future__ import annotations

from bongard.runtime_source_snapshot import capture_loaded_source, verify_loaded_source


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)

from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from bongard import prototype_object_hypotheses as _hypotheses
from bongard import prototype_scene_observer as _scene_runtime
from bongard import visual_witnesses as _visual
from bongard.canonical import canonical_digest, canonical_json
from bongard.evidence import Disposition
from bongard.prototype_object_hypotheses import (
    ObjectHypothesis,
    ObjectHypothesisPacket,
    extract_object_hypothesis_packet,
    object_hypothesis_extractor_artifact_digest,
)
from bongard.prototype_object_lineages import (
    ObjectLineage,
    ObjectLineagePacket,
    extract_object_lineage_packet,
    object_lineage_artifact_digest,
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
    run_codex_named_images_structured,
    validate_codex_named_image_receipt,
    validate_codex_strict_output_schema,
)
from bongard.visual_witnesses import Q16BBox


OBJECT_SCENE_INVENTORY_SCHEMA = "gkm.object-scene-proposal-inventory.v1"
OBJECT_SCENE_CROP_RECEIPT_SCHEMA = "gkm.object-scene-crop-receipt.v1"
OBJECT_SCENE_ATLAS_SHEET_SCHEMA = "gkm.object-scene-atlas-sheet.v1"
OBJECT_SCENE_TRANSCRIPT_SCHEMA = "gkm.object-scene-transcript.v6"
OBJECT_SCENE_TRANSCRIPT_ARTIFACT_SCHEMA = "gkm.object-scene-transcript-artifact.v6"
OBJECT_SCENE_FRONTEND_ID = "object-scene-visual-frontend/stable-lineages-panel-entity-one-call-v6"
OBJECT_SCENE_DROPPED_OPEN_TAG_SCHEMA = "gkm.object-scene-dropped-open-tag.v1"
OBJECT_SCENE_CANONICAL_SCENARIO_ID = "threshold064.close-cross-1"
OBJECT_SCENE_QUALITATIVE_OBSERVABLE_IDS = (
    "triangle_like",
    "quadrilateral_like",
    "sector_like",
    "bird_like",
    "open_contour",
    "closed_boundary",
    "pointed",
    "thin_elongated",
    "necked",
    "mismatched_parts",
    "unequal_part_sizes",
    "unequal_edge_lengths",
    "reflection_symmetry",
    "bilateral_layout",
    "oblique",
    "parallel",
    "perpendicular",
    "crossing",
    "internal_marks",
    "paired_sector_mismatch",
    "triangle_with_three_lines",
)
OBJECT_SCENE_COUNT_OBSERVABLE_IDS = (
    "straight_segment_count",
    "curved_segment_count",
    "endpoint_count",
    "junction_count",
    "closed_loop_count",
    "internal_mark_count",
    "connected_part_count",
    "acute_angle_count",
    "obtuse_angle_count",
    "right_angle_count",
)
OBJECT_SCENE_MAX_TAGS_PER_OBJECT = 8
OBJECT_SCENE_MAX_TAGS_PER_PANEL = 8
OBJECT_SCENE_MAX_TAG_CHARACTERS = 80
OBJECT_SCENE_OPERATIONAL_WITNESS_KINDS = (
    "part_topology",
    "spatial_relation",
    "shape_appearance",
    "marking_pattern",
    "count_relation",
)
OBJECT_SCENE_MIN_REQUIRED_WITNESSES = 1
OBJECT_SCENE_MAX_REQUIRED_WITNESSES = 3
OBJECT_SCENE_MAX_ACCEPTED_VARIANTS = 2
OBJECT_SCENE_MAX_NEAR_MISS_BOUNDARIES = 2
OBJECT_SCENE_MIN_CRITERION_CHARACTERS = 8
OBJECT_SCENE_MAX_CRITERION_CHARACTERS = 160
OBJECT_SCENE_MAX_REGISTERED_TAGS = 32
OBJECT_SCENE_MIN_TAG_PANEL_FREQUENCY = 2
OBJECT_SCENE_SOFT_TAG_SCOPES = ("entity", "panel")
OBJECT_SCENE_SOFT_TAG_ORIENTATION_CONSTRAINTS = (
    "group0_positive",
    "group1_positive",
    "bidirectional",
)

_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_ADDRESS = re.compile(r"sha256:[0-9a-f]{64}\Z")
_OBJECT_ID = re.compile(r"object_[0-9]{4}\Z")
_LINEAGE_ID = re.compile(r"lineage-[0-9]{8}\Z")
_HYPOTHESIS_ID = re.compile(r"hypothesis-[0-9]{8}\Z")
_TAG_ID = re.compile(r"tag_[0-9]{4}\Z")
_WITNESS_ID = re.compile(r"witness_[0-9]{2}\Z")
_ATLAS_NAME = re.compile(r"objects_[0-9]{3}\.png\Z")
_CODE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")
_POSITIVE_TAG = re.compile(
    rf"[a-z][a-z' -]{{1,{OBJECT_SCENE_MAX_TAG_CHARACTERS - 1}}}\Z"
)
_OPERATIONAL_CRITERION = re.compile(
    rf"[a-z][a-z' -]{{{OBJECT_SCENE_MIN_CRITERION_CHARACTERS - 1},"
    rf"{OBJECT_SCENE_MAX_CRITERION_CHARACTERS - 1}}}\Z"
)
_ACCEPTED_VARIANT_CRITERION = re.compile(
    rf"[a-z][a-z', -]{{{OBJECT_SCENE_MIN_CRITERION_CHARACTERS - 1},"
    rf"{OBJECT_SCENE_MAX_CRITERION_CHARACTERS - 1}}}\Z"
)
_NONCANONICAL_CRITERION_COMMA = re.compile(r"\s,|,(?:\S| {2,})|,\Z")
_PROSE = re.compile(r"[ -~]+\Z")
_FORBIDDEN_VISIBLE = re.compile(
    r"\b(?:(?:candidate|group|class|label|target|foil|positive|negative|"
    r"predicate|formula|task|query|answer|prompt|instruction|system|assistant|"
    r"role|bucket|"
    r"user|tool|code|python|theorem|proof)s?)\b",
    re.IGNORECASE,
)
_FORBIDDEN_TAG_LOGIC = re.compile(
    r"\b(?:no|not|none|neither|without|lacks?|lacking|absent|missing|"
    r"except|or|either|versus|unlike)\b",
    re.IGNORECASE,
)
_FORBIDDEN_OPERATIONAL_CARD_LEAK = re.compile(
    r"\b(?:citations?|orientations?|roles?|experiments?|frequency|frequencies|"
    r"frequent|recurring|dataset|training|side[01])\b",
    re.IGNORECASE,
)
_FORBIDDEN_ACCEPTED_VARIANT_LOGIC = re.compile(
    r"\b(?:no|not|none|neither|without|lacks?|lacking|absent|missing|"
    r"except|either|versus|unlike)\b",
    re.IGNORECASE,
)
_FORBIDDEN_BOUNDARY_LOGIC = re.compile(
    r"\b(?:no|not|none|neither|never|without|lacks?|lacking|absent|missing|"
    r"except|or|either|versus|unlike)\b",
    re.IGNORECASE,
)
_EXPLICIT_BOUNDARY_EXCLUSION = re.compile(
    r"\b(?:does not qualify|do not qualify|is excluded|are excluded|"
    r"falls outside|fall outside)\b",
    re.IGNORECASE,
)
OBJECT_SCENE_DISCOVERY_OPEN_TAG_DROP_REASON_CODES = (
    "malformed_item",
    "invalid_tag",
    "invalid_state",
    "invalid_evidence",
    "duplicate_normalized_tag",
    "capacity_exceeded",
)
_UNICODE_DASH = re.compile(r"[ \t]*[\u2013\u2014][ \t]*")
_UNICODE_VISIBLE_TRANSLATION = str.maketrans(
    {
        "\u00a0": " ",
        "\u2010": "-",
        "\u2011": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2026": "...",
        "\u2212": "-",
    }
)


class ObjectSceneVisualFrontendError(ValueError):
    """A deterministic inventory, transcript, or replay is malformed."""


@dataclass(frozen=True, order=True, slots=True)
class UnitSupportInterval:
    """Closed Boolean-support interval; no pseudo-probability is claimed."""

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
            raise ObjectSceneVisualFrontendError("unit support interval differs")

    @classmethod
    def from_state(cls, value: object) -> "UnitSupportInterval":
        try:
            return {
                "present": cls(1, 1),
                "absent": cls(0, 0),
                "indeterminate": cls(0, 1),
            }[value]  # type: ignore[index]
        except (KeyError, TypeError) as exc:
            raise ObjectSceneVisualFrontendError("qualitative support state differs") from exc

    def to_data(self) -> dict[str, int]:
        return {"lower": self.lower, "upper": self.upper}

    @classmethod
    def from_data(cls, value: object) -> "UnitSupportInterval":
        raw = _fields(value, {"lower", "upper"}, "unit support interval")
        return cls(raw["lower"], raw["upper"])


def _authority_data() -> dict[str, object]:
    return {
        "predicate_authority_id": PYTHON_PREDICATE_AUTHORITY_ID,
        "python_is_canonical_authority": True,
        "predicate_input_accepted": False,
        "inventory_frozen_before_transcript": True,
        "semantic_object_completeness_claimed": False,
        "overlap_graph_persisted": True,
        "qualitative_support_intervals": {
            "present": [1, 1],
            "absent": [0, 0],
            "indeterminate": [0, 1],
        },
        "lean_present": False,
        "lean_required": False,
        "lean_removable": True,
        "lean_affects_identity_or_decision": False,
        "lean_required_for_replay": False,
    }


def _fields(value: object, expected: set[str], label: str) -> Mapping[str, Any]:
    if (
        not isinstance(value, Mapping)
        or any(not isinstance(key, str) for key in value)
        or set(value) != expected
    ):
        raise ObjectSceneVisualFrontendError(f"{label} fields differ")
    return value


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ObjectSceneVisualFrontendError(
            f"{label} must be a raw lowercase SHA-256"
        )
    return value


def _address(value: object, label: str) -> str:
    if not isinstance(value, str) or _ADDRESS.fullmatch(value) is None:
        raise ObjectSceneVisualFrontendError(f"{label} must be a sha256: address")
    return value


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ObjectSceneVisualFrontendError(f"{label} must be an integer >= {minimum}")
    return value


def _soft_tag_scope(value: object, label: str = "soft tag scope") -> str:
    if (
        not isinstance(value, str)
        or value not in OBJECT_SCENE_SOFT_TAG_SCOPES
    ):
        raise ObjectSceneVisualFrontendError(
            f"{label} must be one of {OBJECT_SCENE_SOFT_TAG_SCOPES}"
        )
    return value


def _soft_tag_orientation_constraint(
    value: object,
    label: str = "soft tag orientation constraint",
) -> str:
    if (
        not isinstance(value, str)
        or value not in OBJECT_SCENE_SOFT_TAG_ORIENTATION_CONSTRAINTS
    ):
        raise ObjectSceneVisualFrontendError(
            f"{label} must be one of "
            f"{OBJECT_SCENE_SOFT_TAG_ORIENTATION_CONSTRAINTS}"
        )
    return value


def _normalized_operational_clause(
    value: object,
    *,
    label: str,
    allow_canonical_commas: bool = False,
) -> str:
    normalized = _normalize_model_visible_text(value)
    criterion = (
        _ACCEPTED_VARIANT_CRITERION
        if allow_canonical_commas
        else _OPERATIONAL_CRITERION
    )
    if (
        not isinstance(normalized, str)
        or normalized != normalized.strip()
        or "  " in normalized
        or criterion.fullmatch(normalized) is None
        or (
            allow_canonical_commas
            and _NONCANONICAL_CRITERION_COMMA.search(normalized) is not None
        )
        or _FORBIDDEN_VISIBLE.search(normalized) is not None
        or _FORBIDDEN_OPERATIONAL_CARD_LEAK.search(normalized) is not None
    ):
        raise ObjectSceneVisualFrontendError(
            f"{label} is not image-local operational prose"
        )
    return normalized


def normalize_object_scene_witness_statement(value: object) -> str:
    """Normalize one affirmative, independently judgeable image witness."""

    normalized = _normalized_operational_clause(
        value, label="operational witness statement"
    )
    if _FORBIDDEN_TAG_LOGIC.search(normalized) is not None:
        raise ObjectSceneVisualFrontendError(
            "operational witness statement is not affirmative atomic prose"
        )
    return normalized


def normalize_object_scene_accepted_variant(value: object) -> str:
    """Normalize affirmative inclusion/equivalence guidance."""

    normalized = _normalized_operational_clause(
        value,
        label="accepted variant",
        allow_canonical_commas=True,
    )
    if _FORBIDDEN_ACCEPTED_VARIANT_LOGIC.search(normalized) is not None:
        raise ObjectSceneVisualFrontendError(
            "accepted variant is not affirmative inclusion guidance"
        )
    return normalized


def normalize_object_scene_near_miss_boundary(value: object) -> str:
    """Normalize one explicit, controlled exclusion boundary."""

    normalized = _normalized_operational_clause(
        value, label="near-miss boundary"
    )
    exclusions = tuple(_EXPLICIT_BOUNDARY_EXCLUSION.finditer(normalized))
    affirmative_antecedent = (
        normalized
        if len(exclusions) != 1
        else (
            normalized[: exclusions[0].start()]
            + " qualifies"
            + normalized[exclusions[0].end() :]
        )
    )
    if len(exclusions) != 1 or _FORBIDDEN_BOUNDARY_LOGIC.search(
        affirmative_antecedent
    ) is not None:
        raise ObjectSceneVisualFrontendError(
            "near-miss boundary is not an explicit controlled exclusion"
        )
    return normalized


def _canonical_operational_criteria(
    values: object,
    *,
    label: str,
    minimum: int,
    maximum: int,
    normalizer: Any,
) -> tuple[str, ...]:
    if (
        isinstance(values, (str, bytes))
        or not isinstance(values, Sequence)
    ):
        raise ObjectSceneVisualFrontendError(
            f"{label} must be a finite sequence"
        )
    normalized = tuple(normalizer(item) for item in values)
    canonical = tuple(sorted(set(normalized)))
    if not minimum <= len(canonical) <= maximum or len(canonical) != len(normalized):
        raise ObjectSceneVisualFrontendError(f"{label} bounds or uniqueness differ")
    return canonical


def _operational_witness_content(
    value: "ObjectSceneOperationalWitness",
) -> dict[str, object]:
    return {
        "witness_id": value.witness_id,
        "kind": value.kind,
        "statement": value.statement,
    }


@dataclass(frozen=True, order=True, slots=True)
class ObjectSceneOperationalWitness:
    witness_id: str
    kind: str
    statement: str
    witness_digest: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.witness_id, str)
            or _WITNESS_ID.fullmatch(self.witness_id) is None
        ):
            raise ObjectSceneVisualFrontendError("operational witness ID differs")
        if self.kind not in OBJECT_SCENE_OPERATIONAL_WITNESS_KINDS:
            raise ObjectSceneVisualFrontendError("operational witness kind differs")
        if normalize_object_scene_witness_statement(self.statement) != self.statement:
            raise ObjectSceneVisualFrontendError(
                "operational witness statement is not canonical"
            )
        _digest(self.witness_digest, "operational witness digest")
        if self.witness_digest != canonical_digest(
            _operational_witness_content(self)
        ):
            raise ObjectSceneVisualFrontendError(
                "operational witness content digest differs"
            )

    @classmethod
    def create(
        cls,
        witness_id: str,
        kind: str,
        statement: object,
    ) -> "ObjectSceneOperationalWitness":
        normalized = normalize_object_scene_witness_statement(statement)
        provisional = object.__new__(cls)
        for name, item in (
            ("witness_id", witness_id),
            ("kind", kind),
            ("statement", normalized),
        ):
            object.__setattr__(provisional, name, item)
        return cls(
            witness_id,
            kind,
            normalized,
            canonical_digest(_operational_witness_content(provisional)),
        )

    def to_data(self) -> dict[str, object]:
        return {
            **_operational_witness_content(self),
            "witness_digest": self.witness_digest,
        }

    @classmethod
    def from_data(cls, value: object) -> "ObjectSceneOperationalWitness":
        raw = _fields(
            value,
            {"witness_id", "kind", "statement", "witness_digest"},
            "operational witness",
        )
        result = cls(
            raw["witness_id"],
            raw["kind"],
            raw["statement"],
            raw["witness_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectSceneVisualFrontendError(
                "operational witness is not canonical"
            )
        return result


def canonicalize_object_scene_operational_witnesses(
    value: object,
) -> tuple[ObjectSceneOperationalWitness, ...]:
    """Sort unique proposer witnesses, then assign stable local IDs."""

    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ObjectSceneVisualFrontendError(
            "required witnesses must be a finite sequence"
        )
    pairs: list[tuple[str, str]] = []
    for item in value:
        if isinstance(item, ObjectSceneOperationalWitness):
            kind, statement = item.kind, item.statement
        else:
            raw = _fields(
                item,
                {"kind", "statement"},
                "required witness specification",
            )
            kind, statement = raw["kind"], raw["statement"]
        if kind not in OBJECT_SCENE_OPERATIONAL_WITNESS_KINDS:
            raise ObjectSceneVisualFrontendError("operational witness kind differs")
        pairs.append((kind, normalize_object_scene_witness_statement(statement)))
    canonical = sorted(set(pairs))
    if (
        not OBJECT_SCENE_MIN_REQUIRED_WITNESSES
        <= len(canonical)
        <= OBJECT_SCENE_MAX_REQUIRED_WITNESSES
        or len(canonical) != len(pairs)
    ):
        raise ObjectSceneVisualFrontendError(
            "required witness bounds or uniqueness differ"
        )
    return tuple(
        ObjectSceneOperationalWitness.create(
            f"witness_{index:02d}", kind, statement
        )
        for index, (kind, statement) in enumerate(canonical)
    )


def object_scene_soft_tag_criteria_digest(
    required_witnesses: object,
    accepted_variants: object,
    near_miss_boundaries: object,
) -> str:
    witnesses = canonicalize_object_scene_operational_witnesses(
        required_witnesses
    )
    variants = _canonical_operational_criteria(
        accepted_variants,
        label="accepted variants",
        minimum=0,
        maximum=OBJECT_SCENE_MAX_ACCEPTED_VARIANTS,
        normalizer=normalize_object_scene_accepted_variant,
    )
    boundaries = _canonical_operational_criteria(
        near_miss_boundaries,
        label="near-miss boundaries",
        minimum=0,
        maximum=OBJECT_SCENE_MAX_NEAR_MISS_BOUNDARIES,
        normalizer=normalize_object_scene_near_miss_boundary,
    )
    return canonical_digest(
        {
            "required_witnesses": [item.to_data() for item in witnesses],
            "accepted_variants": list(variants),
            "near_miss_boundaries": list(boundaries),
        }
    )


def _soft_tag_content_digest(
    scope: str,
    tag: str,
    criteria_digest: str,
    orientation_constraint: str,
) -> str:
    return canonical_digest(
        {
            "scope": scope,
            "normalized_affirmative_tag": tag,
            "criteria_digest": criteria_digest,
            "orientation_constraint": orientation_constraint,
        }
    )


def _dropped_soft_tag_content_digest(scope: str, tag: str) -> str:
    return canonical_digest(
        {"scope": scope, "normalized_affirmative_tag": tag}
    )


def _operational_card_policy() -> dict[str, object]:
    return {
        "required_witness_count": [
            OBJECT_SCENE_MIN_REQUIRED_WITNESSES,
            OBJECT_SCENE_MAX_REQUIRED_WITNESSES,
        ],
        "required_witness_kinds": list(
            OBJECT_SCENE_OPERATIONAL_WITNESS_KINDS
        ),
        "accepted_variant_count": [0, OBJECT_SCENE_MAX_ACCEPTED_VARIANTS],
        "near_miss_boundary_count": [
            0,
            OBJECT_SCENE_MAX_NEAR_MISS_BOUNDARIES,
        ],
        "criterion_character_count": [
            OBJECT_SCENE_MIN_CRITERION_CHARACTERS,
            OBJECT_SCENE_MAX_CRITERION_CHARACTERS,
        ],
        "required_witness_order": (
            "unique-canonical-kind-statement-then-local-witness-id"
        ),
        "accepted_variant_semantics": (
            "affirmative-inclusion-equivalence-guidance-never-votes"
        ),
        "near_miss_boundary_semantics": (
            "explicit-exclusion-guidance-never-votes"
        ),
        "scope_binding": {
            "panel": "whole-panel",
            "entity": "one-frozen-proposal",
        },
        "disposition_rule": (
            "error-dominant;else-absent-if-any-witness-absent;"
            "else-present-if-all-witnesses-present;else-indeterminate"
        ),
        "model_returns_macro_state": False,
    }


def _bounded_prose(value: object, label: str, maximum: int) -> str:
    if (
        not isinstance(value, str)
        or not 3 <= len(value) <= maximum
        or value != value.strip()
        or "  " in value
        or _PROSE.fullmatch(value) is None
        or _FORBIDDEN_VISIBLE.search(value) is not None
    ):
        raise ObjectSceneVisualFrontendError(f"{label} violates neutral prose policy")
    return value


def _normalize_model_visible_text(value: object) -> object:
    """Normalize a closed set of ordinary model punctuation before parsing."""

    if not isinstance(value, str):
        return value
    normalized = _UNICODE_DASH.sub(" - ", value)
    return normalized.translate(_UNICODE_VISIBLE_TRANSLATION)


def _normalized_bounded_prose(value: object, label: str, maximum: int) -> str:
    return _bounded_prose(_normalize_model_visible_text(value), label, maximum)


def _positive_tag(value: object) -> str:
    if (
        not isinstance(value, str)
        or value != value.strip()
        or "  " in value
        or _POSITIVE_TAG.fullmatch(value) is None
        or _FORBIDDEN_VISIBLE.search(value) is not None
        or _FORBIDDEN_TAG_LOGIC.search(value) is not None
    ):
        raise ObjectSceneVisualFrontendError("open visual tag is not atomic affirmative prose")
    return value


def _normalized_positive_tag(value: object) -> str:
    return _positive_tag(_normalize_model_visible_text(value))


def _canonical_payload(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(k, str) for k in value):
        raise ObjectSceneVisualFrontendError("transcript payload must be an object")
    try:
        result = json.loads(canonical_json(dict(value)).decode("utf-8"))
    except (TypeError, ValueError, UnicodeError, json.JSONDecodeError) as exc:
        raise ObjectSceneVisualFrontendError("transcript payload is not canonical JSON") from exc
    if not isinstance(result, dict):
        raise ObjectSceneVisualFrontendError("transcript payload must be an object")
    return result


def object_scene_visual_frontend_source_digest() -> str:
    return verify_loaded_source(__name__, expected_source_sha256=_LOADED_SOURCE_SHA256)


def object_scene_inventory_protocol_digest() -> str:
    return canonical_digest(
        {
            "schema": "gkm.object-scene-inventory-protocol.v1",
            "frontend_id": OBJECT_SCENE_FRONTEND_ID,
            "source_digest": object_scene_visual_frontend_source_digest(),
            "visual_witness_extractor_digest": _visual.visual_witness_extractor_digest(),
            "hypothesis_extractor_digest": object_hypothesis_extractor_artifact_digest(),
            "lineage_extractor_digest": object_lineage_artifact_digest(),
            "canonical_scenario_id": OBJECT_SCENE_CANONICAL_SCENARIO_ID,
            "proposal_rule": "all-lineages-eligible-for-aggregation",
            "crop_rule": "canonical-scenario-exact-masked-strength",
            "atlas_renderer": "prototype-object-hypotheses-font-free-v1",
            "atlas_capacity": _hypotheses.ATLAS_SLOT_CAPACITY,
            "semantic_object_claimed": False,
            **_authority_data(),
        }
    )


def _sheet_content(value: "ObjectSceneAtlasSheet") -> dict[str, object]:
    return {
        "schema": OBJECT_SCENE_ATLAS_SHEET_SCHEMA,
        "sheet_index": value.sheet_index,
        "name": value.name,
        "width_pixels": value.width_pixels,
        "height_pixels": value.height_pixels,
        "object_ids": list(value.object_ids),
        "png_byte_count": value.png_byte_count,
        "png_digest": value.png_digest,
    }


@dataclass(frozen=True, slots=True)
class ObjectSceneAtlasSheet:
    sheet_index: int
    name: str
    width_pixels: int
    height_pixels: int
    object_ids: tuple[str, ...]
    png_byte_count: int
    png_digest: str
    sheet_digest: str

    def __post_init__(self) -> None:
        _integer(self.sheet_index, "atlas sheet index")
        if not isinstance(self.name, str) or _ATLAS_NAME.fullmatch(self.name) is None:
            raise ObjectSceneVisualFrontendError("atlas sheet name differs")
        if self.name != f"objects_{self.sheet_index:03d}.png":
            raise ObjectSceneVisualFrontendError("atlas sheet index/name differ")
        if (
            self.width_pixels != _hypotheses.ATLAS_WIDTH_PIXELS
            or self.height_pixels != _hypotheses.ATLAS_HEIGHT_PIXELS
        ):
            raise ObjectSceneVisualFrontendError("atlas dimensions differ")
        if (
            not isinstance(self.object_ids, tuple)
            or not 0 < len(self.object_ids) <= _hypotheses.ATLAS_SLOT_CAPACITY
            or any(_OBJECT_ID.fullmatch(item) is None for item in self.object_ids)
        ):
            raise ObjectSceneVisualFrontendError("atlas object inventory differs")
        _integer(self.png_byte_count, "atlas PNG byte count", minimum=1)
        _digest(self.png_digest, "atlas PNG digest")
        _digest(self.sheet_digest, "atlas sheet digest")
        if self.sheet_digest != canonical_digest(_sheet_content(self)):
            raise ObjectSceneVisualFrontendError("atlas sheet digest differs")

    def to_data(self) -> dict[str, object]:
        return {**_sheet_content(self), "sheet_digest": self.sheet_digest}

    @classmethod
    def from_data(cls, value: object) -> "ObjectSceneAtlasSheet":
        raw = _fields(
            value,
            {
                "schema", "sheet_index", "name", "width_pixels", "height_pixels",
                "object_ids", "png_byte_count", "png_digest", "sheet_digest",
            },
            "atlas sheet",
        )
        if raw["schema"] != OBJECT_SCENE_ATLAS_SHEET_SCHEMA or not isinstance(
            raw["object_ids"], list
        ):
            raise ObjectSceneVisualFrontendError("atlas sheet policy differs")
        result = cls(
            raw["sheet_index"], raw["name"], raw["width_pixels"],
            raw["height_pixels"], tuple(raw["object_ids"]), raw["png_byte_count"],
            raw["png_digest"], raw["sheet_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectSceneVisualFrontendError("atlas sheet is not canonical")
        return result


def _crop_content(value: "ObjectSceneCropReceipt") -> dict[str, object]:
    return {
        "schema": OBJECT_SCENE_CROP_RECEIPT_SCHEMA,
        "object_id": value.object_id,
        "lineage_id": value.lineage_id,
        "lineage_digest": value.lineage_digest,
        "scenario_id": value.scenario_id,
        "hypothesis_id": value.hypothesis_id,
        "hypothesis_digest": value.hypothesis_digest,
        "source_component_ids": list(value.source_component_ids),
        "bbox_pixels": list(value.bbox_pixels),
        "bbox_q16": value.bbox_q16.to_data(),
        "union_area_pixels": value.union_area_pixels,
        "emergence_gap_pixels": value.emergence_gap_pixels,
        "masked_crop_pixel_digest": value.masked_crop_pixel_digest,
        "atlas_name": value.atlas_name,
        "sheet_index": value.sheet_index,
        "row_index": value.row_index,
        "column_index": value.column_index,
        "atlas_png_digest": value.atlas_png_digest,
        "overlap_object_ids": list(value.overlap_object_ids),
        "proposal_not_semantic_object": True,
    }


@dataclass(frozen=True, slots=True)
class ObjectSceneCropReceipt:
    object_id: str
    lineage_id: str
    lineage_digest: str
    scenario_id: str
    hypothesis_id: str
    hypothesis_digest: str
    source_component_ids: tuple[str, ...]
    bbox_pixels: tuple[int, int, int, int]
    bbox_q16: Q16BBox
    union_area_pixels: int
    emergence_gap_pixels: int
    masked_crop_pixel_digest: str
    atlas_name: str
    sheet_index: int
    row_index: int
    column_index: int
    atlas_png_digest: str
    overlap_object_ids: tuple[str, ...]
    receipt_digest: str

    @property
    def component_count(self) -> int:
        return len(self.source_component_ids)

    def geometry_cells(self) -> dict[str, object]:
        """Exact Python cells available without interpreting vision prose."""

        return {
            "bbox_pixels": list(self.bbox_pixels),
            "bbox_q16": self.bbox_q16.to_data(),
            "area_pixels": self.union_area_pixels,
            "component_count": self.component_count,
            "emergence_gap_pixels": self.emergence_gap_pixels,
            "overlap_object_ids": list(self.overlap_object_ids),
        }

    def __post_init__(self) -> None:
        if not isinstance(self.object_id, str) or _OBJECT_ID.fullmatch(self.object_id) is None:
            raise ObjectSceneVisualFrontendError("object ID differs")
        if not isinstance(self.lineage_id, str) or _LINEAGE_ID.fullmatch(self.lineage_id) is None:
            raise ObjectSceneVisualFrontendError("lineage ID differs")
        _digest(self.lineage_digest, "lineage digest")
        if self.scenario_id != OBJECT_SCENE_CANONICAL_SCENARIO_ID:
            raise ObjectSceneVisualFrontendError("crop scenario differs")
        if not isinstance(self.hypothesis_id, str) or _HYPOTHESIS_ID.fullmatch(self.hypothesis_id) is None:
            raise ObjectSceneVisualFrontendError("hypothesis ID differs")
        _digest(self.hypothesis_digest, "hypothesis digest")
        if (
            not isinstance(self.source_component_ids, tuple)
            or not self.source_component_ids
            or self.source_component_ids != tuple(sorted(set(self.source_component_ids)))
        ):
            raise ObjectSceneVisualFrontendError("crop component IDs differ")
        if (
            not isinstance(self.bbox_pixels, tuple)
            or len(self.bbox_pixels) != 4
            or any(type(item) is not int for item in self.bbox_pixels)
        ):
            raise ObjectSceneVisualFrontendError("crop pixel bbox differs")
        x0, y0, x1, y1 = self.bbox_pixels
        if min(x0, y0) < 0 or x1 <= x0 or y1 <= y0:
            raise ObjectSceneVisualFrontendError("crop bbox has no positive extent")
        if not isinstance(self.bbox_q16, Q16BBox):
            raise TypeError("crop bbox_q16 must be Q16BBox")
        _integer(self.union_area_pixels, "crop union area", minimum=1)
        _integer(self.emergence_gap_pixels, "crop emergence gap")
        _digest(self.masked_crop_pixel_digest, "masked crop pixel digest")
        if not isinstance(self.atlas_name, str) or _ATLAS_NAME.fullmatch(self.atlas_name) is None:
            raise ObjectSceneVisualFrontendError("crop atlas name differs")
        for label, item in (
            ("sheet index", self.sheet_index),
            ("row index", self.row_index),
            ("column index", self.column_index),
        ):
            _integer(item, label)
        if (
            self.atlas_name != f"objects_{self.sheet_index:03d}.png"
            or self.row_index >= _hypotheses.ATLAS_ROWS
            or self.column_index >= _hypotheses.ATLAS_COLUMNS
        ):
            raise ObjectSceneVisualFrontendError("crop atlas slot differs")
        _digest(self.atlas_png_digest, "crop atlas PNG digest")
        if (
            not isinstance(self.overlap_object_ids, tuple)
            or self.overlap_object_ids != tuple(sorted(set(self.overlap_object_ids)))
            or self.object_id in self.overlap_object_ids
            or any(_OBJECT_ID.fullmatch(item) is None for item in self.overlap_object_ids)
        ):
            raise ObjectSceneVisualFrontendError("crop overlap IDs differ")
        _digest(self.receipt_digest, "crop receipt digest")
        if self.receipt_digest != canonical_digest(_crop_content(self)):
            raise ObjectSceneVisualFrontendError("crop receipt digest differs")

    def to_data(self) -> dict[str, object]:
        return {**_crop_content(self), "receipt_digest": self.receipt_digest}

    @classmethod
    def from_data(cls, value: object) -> "ObjectSceneCropReceipt":
        raw = _fields(
            value,
            {
                "schema", "object_id", "lineage_id", "lineage_digest", "scenario_id",
                "hypothesis_id", "hypothesis_digest", "source_component_ids",
                "bbox_pixels", "bbox_q16", "union_area_pixels", "emergence_gap_pixels",
                "masked_crop_pixel_digest", "atlas_name", "sheet_index", "row_index",
                "column_index", "atlas_png_digest", "overlap_object_ids",
                "proposal_not_semantic_object", "receipt_digest",
            },
            "crop receipt",
        )
        if (
            raw["schema"] != OBJECT_SCENE_CROP_RECEIPT_SCHEMA
            or raw["proposal_not_semantic_object"] is not True
            or not isinstance(raw["source_component_ids"], list)
            or not isinstance(raw["bbox_pixels"], list)
            or not isinstance(raw["bbox_q16"], Mapping)
            or not isinstance(raw["overlap_object_ids"], list)
        ):
            raise ObjectSceneVisualFrontendError("crop receipt policy differs")
        result = cls(
            raw["object_id"], raw["lineage_id"], raw["lineage_digest"],
            raw["scenario_id"], raw["hypothesis_id"], raw["hypothesis_digest"],
            tuple(raw["source_component_ids"]), tuple(raw["bbox_pixels"]),
            Q16BBox.from_data(raw["bbox_q16"]), raw["union_area_pixels"],
            raw["emergence_gap_pixels"], raw["masked_crop_pixel_digest"],
            raw["atlas_name"], raw["sheet_index"], raw["row_index"],
            raw["column_index"], raw["atlas_png_digest"],
            tuple(raw["overlap_object_ids"]), raw["receipt_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectSceneVisualFrontendError("crop receipt is not canonical")
        return result


def _inventory_content(value: "ObjectSceneProposalInventory") -> dict[str, object]:
    return {
        "schema": OBJECT_SCENE_INVENTORY_SCHEMA,
        "frontend_id": OBJECT_SCENE_FRONTEND_ID,
        "panel_digest": value.panel_digest,
        "width_pixels": value.width_pixels,
        "height_pixels": value.height_pixels,
        "visual_witness_packet_digest": value.visual_witness_packet_digest,
        "hypothesis_packet_digest": value.hypothesis_packet_digest,
        "lineage_packet": value.lineage_packet.to_data(),
        "lineage_packet_digest": value.lineage_packet_digest,
        "protocol_digest": value.protocol_digest,
        "source_digest": value.source_digest,
        "inventory_status": value.inventory_status,
        "catalog_complete_under_rule": value.catalog_complete_under_rule,
        "diagnostic_codes": list(value.diagnostic_codes),
        "objects": [item.to_data() for item in value.objects],
        "atlas_sheets": [item.to_data() for item in value.atlas_sheets],
        "stable_id_scope": "same-exact-PNG-and-extractor-sources",
        "proposal_semantics": "stable-cross-scenario-raster-lineage-not-semantic-object",
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectSceneProposalInventory:
    panel_digest: str
    width_pixels: int
    height_pixels: int
    visual_witness_packet_digest: str
    hypothesis_packet_digest: str
    lineage_packet: ObjectLineagePacket
    lineage_packet_digest: str
    protocol_digest: str
    source_digest: str
    inventory_status: str
    catalog_complete_under_rule: bool
    diagnostic_codes: tuple[str, ...]
    objects: tuple[ObjectSceneCropReceipt, ...]
    atlas_sheets: tuple[ObjectSceneAtlasSheet, ...]
    inventory_digest: str

    def __post_init__(self) -> None:
        for name in (
            "panel_digest", "visual_witness_packet_digest", "hypothesis_packet_digest",
            "lineage_packet_digest", "protocol_digest", "source_digest", "inventory_digest",
        ):
            _digest(getattr(self, name), name)
        _integer(self.width_pixels, "inventory width", minimum=2)
        _integer(self.height_pixels, "inventory height", minimum=2)
        if not isinstance(self.lineage_packet, ObjectLineagePacket):
            raise TypeError("inventory lineage packet has the wrong type")
        if (
            self.lineage_packet_digest != self.lineage_packet.digest()
            or self.lineage_packet.panel_digest != self.panel_digest
            or self.lineage_packet.hypothesis_packet_digest != self.hypothesis_packet_digest
            or self.protocol_digest != object_scene_inventory_protocol_digest()
            or self.source_digest != object_scene_visual_frontend_source_digest()
        ):
            raise ObjectSceneVisualFrontendError("inventory dependency binding differs")
        if self.inventory_status != "complete" or self.catalog_complete_under_rule is not True:
            raise ObjectSceneVisualFrontendError("inventory status differs")
        if (
            not isinstance(self.diagnostic_codes, tuple)
            or self.diagnostic_codes != tuple(sorted(set(self.diagnostic_codes)))
            or any(_CODE.fullmatch(item) is None for item in self.diagnostic_codes)
        ):
            raise ObjectSceneVisualFrontendError("inventory diagnostics differ")
        if (
            not isinstance(self.objects, tuple)
            or tuple(item.object_id for item in self.objects)
            != tuple(f"object_{index:04d}" for index in range(len(self.objects)))
            or not isinstance(self.atlas_sheets, tuple)
            or tuple(item.sheet_index for item in self.atlas_sheets)
            != tuple(range(len(self.atlas_sheets)))
        ):
            raise ObjectSceneVisualFrontendError("inventory order differs")
        flattened = tuple(item for sheet in self.atlas_sheets for item in sheet.object_ids)
        if flattened != tuple(item.object_id for item in self.objects):
            raise ObjectSceneVisualFrontendError("atlas omits or reorders proposals")
        by_id = {item.object_id: item for item in self.objects}
        by_sheet = {item.name: item for item in self.atlas_sheets}
        for item in self.objects:
            sheet = by_sheet.get(item.atlas_name)
            if (
                sheet is None
                or item.atlas_png_digest != sheet.png_digest
                or sheet.object_ids[item.row_index * _hypotheses.ATLAS_COLUMNS + item.column_index]
                != item.object_id
            ):
                raise ObjectSceneVisualFrontendError("crop receipt/atlas binding differs")
            for other_id in item.overlap_object_ids:
                if item.object_id not in by_id[other_id].overlap_object_ids:
                    raise ObjectSceneVisualFrontendError("overlap graph is not symmetric")
        _digest(self.inventory_digest, "inventory digest")
        if self.inventory_digest != canonical_digest(_inventory_content(self)):
            raise ObjectSceneVisualFrontendError("inventory digest differs")

    def to_data(self) -> dict[str, object]:
        return {**_inventory_content(self), "inventory_digest": self.inventory_digest}

    @classmethod
    def from_data(cls, value: object) -> "ObjectSceneProposalInventory":
        expected = {
            "schema", "frontend_id", "panel_digest", "width_pixels", "height_pixels",
            "visual_witness_packet_digest", "hypothesis_packet_digest", "lineage_packet",
            "lineage_packet_digest", "protocol_digest", "source_digest", "inventory_status",
            "catalog_complete_under_rule", "diagnostic_codes", "objects", "atlas_sheets", "stable_id_scope",
            "proposal_semantics", *_authority_data(), "inventory_digest",
        }
        raw = _fields(value, expected, "proposal inventory")
        if (
            raw["schema"] != OBJECT_SCENE_INVENTORY_SCHEMA
            or raw["frontend_id"] != OBJECT_SCENE_FRONTEND_ID
            or raw["stable_id_scope"] != "same-exact-PNG-and-extractor-sources"
            or raw["proposal_semantics"]
            != "stable-cross-scenario-raster-lineage-not-semantic-object"
            or any(raw[key] != item for key, item in _authority_data().items())
            or raw["catalog_complete_under_rule"] is not True
            or not isinstance(raw["diagnostic_codes"], list)
            or not isinstance(raw["objects"], list)
            or not isinstance(raw["atlas_sheets"], list)
        ):
            raise ObjectSceneVisualFrontendError("proposal inventory policy differs")
        result = cls(
            raw["panel_digest"], raw["width_pixels"], raw["height_pixels"],
            raw["visual_witness_packet_digest"], raw["hypothesis_packet_digest"],
            ObjectLineagePacket.from_data(raw["lineage_packet"]),
            raw["lineage_packet_digest"], raw["protocol_digest"], raw["source_digest"],
            raw["inventory_status"], raw["catalog_complete_under_rule"],
            tuple(raw["diagnostic_codes"]),
            tuple(ObjectSceneCropReceipt.from_data(item) for item in raw["objects"]),
            tuple(ObjectSceneAtlasSheet.from_data(item) for item in raw["atlas_sheets"]),
            raw["inventory_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectSceneVisualFrontendError("proposal inventory is not canonical")
        return result


def _hypothesis_crop_map(
    png_bytes: bytes,
    packet: ObjectHypothesisPacket,
) -> dict[tuple[str, str], np.ndarray]:
    visual = _visual.extract_visual_witnesses(png_bytes)
    if visual.digest() != packet.visual_witness_packet_digest:
        raise ObjectSceneVisualFrontendError("visual packet binding differs")
    strength = _visual._decode_png(png_bytes)
    crops = _hypotheses._crops_for_catalog(strength, visual, packet.scenarios)
    hypotheses = tuple(
        item for scenario in packet.scenarios for item in scenario.hypotheses
    )
    if len(crops) != len(hypotheses):
        raise ObjectSceneVisualFrontendError("hypothesis crop inventory differs")
    return {
        (item.scenario_id, item.hypothesis_id): crop
        for item, crop in zip(hypotheses, crops, strict=True)
    }


def _hypothesis_index(
    packet: ObjectHypothesisPacket,
) -> dict[tuple[str, str], ObjectHypothesis]:
    return {
        (item.scenario_id, item.hypothesis_id): item
        for scenario in packet.scenarios
        for item in scenario.hypotheses
    }


def _make_sheet(
    sheet_index: int,
    object_ids: tuple[str, ...],
    png_bytes: bytes,
) -> ObjectSceneAtlasSheet:
    values = {
        "sheet_index": sheet_index,
        "name": f"objects_{sheet_index:03d}.png",
        "width_pixels": _hypotheses.ATLAS_WIDTH_PIXELS,
        "height_pixels": _hypotheses.ATLAS_HEIGHT_PIXELS,
        "object_ids": object_ids,
        "png_byte_count": len(png_bytes),
        "png_digest": hashlib.sha256(png_bytes).hexdigest(),
    }
    provisional = object.__new__(ObjectSceneAtlasSheet)
    for name, item in values.items():
        object.__setattr__(provisional, name, item)
    return ObjectSceneAtlasSheet(
        **values,
        sheet_digest=canonical_digest(_sheet_content(provisional)),
    )


def _make_crop_receipt(
    *,
    object_id: str,
    lineage: ObjectLineage,
    hypothesis: ObjectHypothesis,
    sheet: ObjectSceneAtlasSheet,
    local_index: int,
    overlap_object_ids: tuple[str, ...],
) -> ObjectSceneCropReceipt:
    values = {
        "object_id": object_id,
        "lineage_id": lineage.lineage_id,
        "lineage_digest": canonical_digest(lineage.to_data()),
        "scenario_id": hypothesis.scenario_id,
        "hypothesis_id": hypothesis.hypothesis_id,
        "hypothesis_digest": hypothesis.digest(),
        "source_component_ids": hypothesis.source_component_ids,
        "bbox_pixels": hypothesis.bbox_pixels,
        "bbox_q16": hypothesis.bbox_q16,
        "union_area_pixels": hypothesis.union_area_pixels,
        "emergence_gap_pixels": hypothesis.emergence_gap_pixels,
        "masked_crop_pixel_digest": hypothesis.masked_crop_pixel_digest,
        "atlas_name": sheet.name,
        "sheet_index": sheet.sheet_index,
        "row_index": local_index // _hypotheses.ATLAS_COLUMNS,
        "column_index": local_index % _hypotheses.ATLAS_COLUMNS,
        "atlas_png_digest": sheet.png_digest,
        "overlap_object_ids": overlap_object_ids,
    }
    provisional = object.__new__(ObjectSceneCropReceipt)
    for name, item in values.items():
        object.__setattr__(provisional, name, item)
    return ObjectSceneCropReceipt(
        **values,
        receipt_digest=canonical_digest(_crop_content(provisional)),
    )


@lru_cache(maxsize=128)
def _build_object_scene_inventory(
    png_bytes: bytes,
) -> tuple[ObjectSceneProposalInventory, tuple[tuple[str, bytes], ...]]:
    if not isinstance(png_bytes, bytes):
        raise TypeError("object-scene inventory input must be exact PNG bytes")
    hypotheses = extract_object_hypothesis_packet(png_bytes)
    lineages = extract_object_lineage_packet(png_bytes, hypotheses)
    eligible = tuple(item for item in lineages.lineages if item.eligible_for_aggregation)
    hypothesis_by_key = _hypothesis_index(hypotheses)
    crop_by_key = _hypothesis_crop_map(png_bytes, hypotheses)
    rows: list[tuple[ObjectLineage, ObjectHypothesis, np.ndarray]] = []
    for lineage in eligible:
        member = next(
            item
            for item in lineage.members
            if item.scenario_id == OBJECT_SCENE_CANONICAL_SCENARIO_ID
        )
        key = (member.scenario_id, member.hypothesis_id)
        hypothesis = hypothesis_by_key[key]
        crop = crop_by_key[key]
        if (
            hypothesis.digest() != member.hypothesis_digest
            or hypothesis.masked_crop_pixel_digest
            != _hypotheses._crop_pixel_digest(crop)
        ):
            raise ObjectSceneVisualFrontendError("canonical crop replay differs")
        rows.append((lineage, hypothesis, crop))

    object_ids = tuple(f"object_{index:04d}" for index in range(len(rows)))
    rendered: list[tuple[str, bytes]] = []
    sheets: list[ObjectSceneAtlasSheet] = []
    capacity = _hypotheses.ATLAS_SLOT_CAPACITY
    for sheet_index, start in enumerate(range(0, len(rows), capacity)):
        chunk = rows[start : start + capacity]
        png = _hypotheses._render_atlas_sheet(tuple(item[2] for item in chunk))
        ids = object_ids[start : start + len(chunk)]
        sheet = _make_sheet(sheet_index, ids, png)
        sheets.append(sheet)
        rendered.append((sheet.name, png))

    component_sets = tuple(frozenset(item[1].source_component_ids) for item in rows)
    overlaps: list[tuple[str, ...]] = []
    for index, components in enumerate(component_sets):
        overlaps.append(
            tuple(
                object_ids[other]
                for other, other_components in enumerate(component_sets)
                if other != index and components & other_components
            )
        )
    objects: list[ObjectSceneCropReceipt] = []
    for index, (lineage, hypothesis, _) in enumerate(rows):
        sheet_index, local_index = divmod(index, capacity)
        objects.append(
            _make_crop_receipt(
                object_id=object_ids[index],
                lineage=lineage,
                hypothesis=hypothesis,
                sheet=sheets[sheet_index],
                local_index=local_index,
                overlap_object_ids=overlaps[index],
            )
        )

    # This status describes completeness of the frozen proposal catalog, not
    # semantic-object completeness.  Overlap is retained in the graph and does
    # not globally poison existential observations.
    reasons: list[str] = []
    if lineages.unlinked_hypothesis_count:
        reasons.append("unlinked_hypotheses")
    if lineages.ambiguous_member_target_count:
        reasons.append("ambiguous_cross_scenario_linkage")
    if not eligible and lineages.hypothesis_count:
        reasons.append("no_stable_proposals_with_foreground")
    frozen_reasons = tuple(sorted(set(reasons)))
    values = {
        "panel_digest": hypotheses.panel_digest,
        "width_pixels": hypotheses.width_pixels,
        "height_pixels": hypotheses.height_pixels,
        "visual_witness_packet_digest": hypotheses.visual_witness_packet_digest,
        "hypothesis_packet_digest": hypotheses.digest(),
        "lineage_packet": lineages,
        "lineage_packet_digest": lineages.digest(),
        "protocol_digest": object_scene_inventory_protocol_digest(),
        "source_digest": object_scene_visual_frontend_source_digest(),
        "inventory_status": "complete",
        "catalog_complete_under_rule": True,
        "diagnostic_codes": frozen_reasons,
        "objects": tuple(objects),
        "atlas_sheets": tuple(sheets),
    }
    provisional = object.__new__(ObjectSceneProposalInventory)
    for name, item in values.items():
        object.__setattr__(provisional, name, item)
    inventory = ObjectSceneProposalInventory(
        **values,
        inventory_digest=canonical_digest(_inventory_content(provisional)),
    )
    return inventory, tuple(rendered)


def extract_object_scene_proposal_inventory(
    png_bytes: bytes,
) -> ObjectSceneProposalInventory:
    """Freeze proposal IDs, geometry, overlap, and atlas receipts from pixels."""

    return _build_object_scene_inventory(png_bytes)[0]


def render_object_scene_proposal_atlas(
    inventory: ObjectSceneProposalInventory,
    png_bytes: bytes,
) -> tuple[tuple[str, bytes], ...]:
    if not isinstance(inventory, ObjectSceneProposalInventory):
        raise TypeError("inventory must be ObjectSceneProposalInventory")
    rebuilt, rendered = _build_object_scene_inventory(png_bytes)
    if rebuilt != inventory:
        raise ObjectSceneVisualFrontendError("inventory differs from exact PNG replay")
    return rendered


def verify_object_scene_proposal_inventory(
    inventory: ObjectSceneProposalInventory,
    png_bytes: bytes,
    *,
    expected_inventory_digest: str | None = None,
    expected_atlas_png_by_name: Mapping[str, bytes] | None = None,
) -> ObjectSceneProposalInventory:
    if not isinstance(inventory, ObjectSceneProposalInventory):
        raise TypeError("inventory must be ObjectSceneProposalInventory")
    restored = ObjectSceneProposalInventory.from_data(inventory.to_data())
    if expected_inventory_digest is not None and restored.inventory_digest != _digest(
        expected_inventory_digest, "expected inventory digest"
    ):
        raise ObjectSceneVisualFrontendError("inventory differs from commitment")
    rebuilt, rendered = _build_object_scene_inventory(png_bytes)
    if rebuilt != restored:
        raise ObjectSceneVisualFrontendError("inventory differs from exact PNG replay")
    if expected_atlas_png_by_name is not None:
        supplied = dict(expected_atlas_png_by_name)
        if supplied != dict(rendered):
            raise ObjectSceneVisualFrontendError("proposal atlas bytes differ")
    return restored


class ObjectSceneTranscriptMode(str, Enum):
    DISCOVERY = "discovery"
    REGISTERED_EVALUATION = "registered_evaluation"


@dataclass(frozen=True, order=True, slots=True)
class CountInterval:
    lower: int
    upper: int

    def __post_init__(self) -> None:
        if (
            type(self.lower) is not int
            or type(self.upper) is not int
            or not 0 <= self.lower <= self.upper <= 999
        ):
            raise ObjectSceneVisualFrontendError("count interval differs")

    def to_data(self) -> dict[str, int]:
        return {"lower": self.lower, "upper": self.upper}

    @classmethod
    def from_data(cls, value: object) -> "CountInterval":
        raw = _fields(value, {"lower", "upper"}, "count interval")
        return cls(raw["lower"], raw["upper"])


def _disposition_from_state(value: object) -> Disposition:
    try:
        return {
            "present": Disposition.PRESENT,
            "absent": Disposition.CERTIFIED_ABSENT,
            "indeterminate": Disposition.INDETERMINATE,
        }[value]  # type: ignore[index]
    except (KeyError, TypeError) as exc:
        raise ObjectSceneVisualFrontendError("qualitative state differs") from exc


def _state_from_disposition(value: Disposition) -> str:
    try:
        return {
            Disposition.PRESENT: "present",
            Disposition.CERTIFIED_ABSENT: "absent",
            Disposition.INDETERMINATE: "indeterminate",
        }[value]
    except KeyError as exc:
        raise ObjectSceneVisualFrontendError("visual cell cannot encode error") from exc


def _qualitative_content(value: "ObjectSceneQualitativeCell") -> dict[str, object]:
    return {
        "observable_id": value.observable_id,
        "state": _state_from_disposition(value.disposition),
        "support": value.support.to_data(),
        "evidence": value.evidence,
    }


@dataclass(frozen=True, slots=True)
class ObjectSceneQualitativeCell:
    observable_id: str
    disposition: Disposition
    support: UnitSupportInterval
    evidence: str
    cell_digest: str

    def __post_init__(self) -> None:
        if self.observable_id not in OBJECT_SCENE_QUALITATIVE_OBSERVABLE_IDS:
            raise ObjectSceneVisualFrontendError("qualitative observable differs")
        state = _state_from_disposition(self.disposition)
        if self.support != UnitSupportInterval.from_state(state):
            raise ObjectSceneVisualFrontendError("qualitative support differs")
        _bounded_prose(self.evidence, "qualitative evidence", 240)
        _digest(self.cell_digest, "qualitative cell digest")
        if self.cell_digest != canonical_digest(_qualitative_content(self)):
            raise ObjectSceneVisualFrontendError("qualitative cell digest differs")

    @classmethod
    def create(cls, observable_id: str, state: object, evidence: object) -> "ObjectSceneQualitativeCell":
        disposition = _disposition_from_state(state)
        support = UnitSupportInterval.from_state(state)
        normalized_evidence = _normalized_bounded_prose(
            evidence, "qualitative evidence", 240
        )
        provisional = object.__new__(cls)
        for name, item in (
            ("observable_id", observable_id), ("disposition", disposition),
            ("support", support), ("evidence", normalized_evidence),
        ):
            object.__setattr__(provisional, name, item)
        return cls(observable_id, disposition, support, normalized_evidence, canonical_digest(_qualitative_content(provisional)))

    def to_data(self) -> dict[str, object]:
        return {**_qualitative_content(self), "cell_digest": self.cell_digest}

    @classmethod
    def from_data(cls, value: object) -> "ObjectSceneQualitativeCell":
        raw = _fields(value, {"observable_id", "state", "support", "evidence", "cell_digest"}, "qualitative cell")
        result = cls(
            raw["observable_id"], _disposition_from_state(raw["state"]),
            UnitSupportInterval.from_data(raw["support"]), raw["evidence"], raw["cell_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectSceneVisualFrontendError("qualitative cell is not canonical")
        return result


def _count_content(value: "ObjectSceneCountCell") -> dict[str, object]:
    return {
        "observable_id": value.observable_id,
        "state": value.state,
        "interval": None if value.interval is None else value.interval.to_data(),
        "evidence": value.evidence,
    }


@dataclass(frozen=True, slots=True)
class ObjectSceneCountCell:
    observable_id: str
    state: str
    interval: CountInterval | None
    evidence: str
    cell_digest: str

    def __post_init__(self) -> None:
        if self.observable_id not in OBJECT_SCENE_COUNT_OBSERVABLE_IDS:
            raise ObjectSceneVisualFrontendError("count observable differs")
        if self.state not in ("measured", "indeterminate"):
            raise ObjectSceneVisualFrontendError("count state differs")
        if (self.state == "measured") != isinstance(self.interval, CountInterval):
            raise ObjectSceneVisualFrontendError("count interval/state differ")
        _bounded_prose(self.evidence, "count evidence", 240)
        _digest(self.cell_digest, "count cell digest")
        if self.cell_digest != canonical_digest(_count_content(self)):
            raise ObjectSceneVisualFrontendError("count cell digest differs")

    @classmethod
    def create(cls, observable_id: str, state: object, lower: object, upper: object, evidence: object) -> "ObjectSceneCountCell":
        if state == "measured":
            interval: CountInterval | None = CountInterval(lower, upper)  # type: ignore[arg-type]
        elif state == "indeterminate" and lower is None and upper is None:
            interval = None
        else:
            raise ObjectSceneVisualFrontendError("count payload state differs")
        normalized_evidence = _normalized_bounded_prose(
            evidence, "count evidence", 240
        )
        provisional = object.__new__(cls)
        for name, item in (("observable_id", observable_id), ("state", state), ("interval", interval), ("evidence", normalized_evidence)):
            object.__setattr__(provisional, name, item)
        return cls(observable_id, state, interval, normalized_evidence, canonical_digest(_count_content(provisional)))

    def to_data(self) -> dict[str, object]:
        return {**_count_content(self), "cell_digest": self.cell_digest}

    @classmethod
    def from_data(cls, value: object) -> "ObjectSceneCountCell":
        raw = _fields(value, {"observable_id", "state", "interval", "evidence", "cell_digest"}, "count cell")
        interval = None if raw["interval"] is None else CountInterval.from_data(raw["interval"])
        result = cls(raw["observable_id"], raw["state"], interval, raw["evidence"], raw["cell_digest"])
        if result.to_data() != dict(raw):
            raise ObjectSceneVisualFrontendError("count cell is not canonical")
        return result


def _open_tag_content(value: "ObjectSceneOpenTag") -> dict[str, object]:
    return {
        "tag": value.tag,
        "state": _state_from_disposition(value.disposition),
        "support": value.support.to_data(),
        "evidence": value.evidence,
    }


@dataclass(frozen=True, slots=True)
class ObjectSceneOpenTag:
    tag: str
    disposition: Disposition
    support: UnitSupportInterval
    evidence: str
    tag_observation_digest: str

    def __post_init__(self) -> None:
        _positive_tag(self.tag)
        if self.disposition not in (Disposition.PRESENT, Disposition.INDETERMINATE):
            raise ObjectSceneVisualFrontendError("discovery tags cannot assert absence")
        state = _state_from_disposition(self.disposition)
        if self.support != UnitSupportInterval.from_state(state):
            raise ObjectSceneVisualFrontendError("open tag support differs")
        _bounded_prose(self.evidence, "open tag evidence", 240)
        _digest(self.tag_observation_digest, "open tag observation digest")
        if self.tag_observation_digest != canonical_digest(_open_tag_content(self)):
            raise ObjectSceneVisualFrontendError("open tag digest differs")

    @classmethod
    def create(cls, tag: object, state: object, evidence: object) -> "ObjectSceneOpenTag":
        phrase = _normalized_positive_tag(tag)
        disposition = _disposition_from_state(state)
        support = UnitSupportInterval.from_state(state)
        normalized_evidence = _normalized_bounded_prose(
            evidence, "open tag evidence", 240
        )
        provisional = object.__new__(cls)
        for name, item in (("tag", phrase), ("disposition", disposition), ("support", support), ("evidence", normalized_evidence)):
            object.__setattr__(provisional, name, item)
        return cls(phrase, disposition, support, normalized_evidence, canonical_digest(_open_tag_content(provisional)))

    def to_data(self) -> dict[str, object]:
        return {**_open_tag_content(self), "tag_observation_digest": self.tag_observation_digest}

    @classmethod
    def from_data(cls, value: object) -> "ObjectSceneOpenTag":
        raw = _fields(value, {"tag", "state", "support", "evidence", "tag_observation_digest"}, "open tag")
        result = cls(raw["tag"], _disposition_from_state(raw["state"]), UnitSupportInterval.from_data(raw["support"]), raw["evidence"], raw["tag_observation_digest"])
        if result.to_data() != dict(raw):
            raise ObjectSceneVisualFrontendError("open tag is not canonical")
        return result


def _dropped_open_tag_content(
    value: "ObjectSceneDroppedOpenTag",
) -> dict[str, object]:
    return {
        "schema": OBJECT_SCENE_DROPPED_OPEN_TAG_SCHEMA,
        "scope": value.scope,
        "owner_id": value.owner_id,
        "item_index": value.item_index,
        "raw_item_digest": value.raw_item_digest,
        "reason_code": value.reason_code,
    }


def _dropped_open_tag_sort_key(
    value: "ObjectSceneDroppedOpenTag",
) -> tuple[int, str, int, str, str]:
    return (
        0 if value.scope == "panel" else 1,
        value.owner_id,
        value.item_index,
        value.raw_item_digest,
        value.reason_code,
    )


@dataclass(frozen=True, slots=True)
class ObjectSceneDroppedOpenTag:
    """One quarantined optional discovery tag, bound to its raw payload item."""

    scope: str
    owner_id: str
    item_index: int
    raw_item_digest: str
    reason_code: str
    drop_digest: str

    def __post_init__(self) -> None:
        _soft_tag_scope(self.scope, "dropped open tag scope")
        if (
            (self.scope == "panel" and self.owner_id != "panel")
            or (
                self.scope == "entity"
                and (
                    not isinstance(self.owner_id, str)
                    or _OBJECT_ID.fullmatch(self.owner_id) is None
                )
            )
        ):
            raise ObjectSceneVisualFrontendError(
                "dropped open tag owner differs"
            )
        _integer(self.item_index, "dropped open tag item index")
        _digest(self.raw_item_digest, "dropped open tag raw item digest")
        if self.reason_code not in OBJECT_SCENE_DISCOVERY_OPEN_TAG_DROP_REASON_CODES:
            raise ObjectSceneVisualFrontendError(
                "dropped open tag reason code differs"
            )
        _digest(self.drop_digest, "dropped open tag digest")
        if self.drop_digest != canonical_digest(_dropped_open_tag_content(self)):
            raise ObjectSceneVisualFrontendError(
                "dropped open tag content digest differs"
            )

    @classmethod
    def create(
        cls,
        *,
        scope: str,
        owner_id: str,
        item_index: int,
        raw_item: object,
        reason_code: str,
    ) -> "ObjectSceneDroppedOpenTag":
        values = {
            "scope": scope,
            "owner_id": owner_id,
            "item_index": item_index,
            "raw_item_digest": canonical_digest(raw_item),
            "reason_code": reason_code,
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(
            **values,
            drop_digest=canonical_digest(_dropped_open_tag_content(provisional)),
        )

    def to_data(self) -> dict[str, object]:
        return {
            **_dropped_open_tag_content(self),
            "drop_digest": self.drop_digest,
        }

    @classmethod
    def from_data(cls, value: object) -> "ObjectSceneDroppedOpenTag":
        raw = _fields(
            value,
            {
                "schema",
                "scope",
                "owner_id",
                "item_index",
                "raw_item_digest",
                "reason_code",
                "drop_digest",
            },
            "dropped open tag",
        )
        if raw["schema"] != OBJECT_SCENE_DROPPED_OPEN_TAG_SCHEMA:
            raise ObjectSceneVisualFrontendError(
                "dropped open tag schema differs"
            )
        result = cls(
            raw["scope"],
            raw["owner_id"],
            raw["item_index"],
            raw["raw_item_digest"],
            raw["reason_code"],
            raw["drop_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectSceneVisualFrontendError(
                "dropped open tag is not canonical"
            )
        return result


def _registered_state_from_disposition(value: Disposition) -> str:
    if value is Disposition.ERROR:
        return "error"
    return _state_from_disposition(value)


def _registered_disposition_from_state(value: object) -> Disposition:
    if value == "error":
        return Disposition.ERROR
    return _disposition_from_state(value)


def _registered_support_from_disposition(
    value: Disposition,
) -> UnitSupportInterval | None:
    if value is Disposition.ERROR:
        return None
    return UnitSupportInterval.from_state(_registered_state_from_disposition(value))


def _registered_witness_cell_content(
    value: "ObjectSceneRegisteredWitnessCell",
) -> dict[str, object]:
    return {
        "witness_id": value.witness_id,
        "witness_digest": value.witness_digest,
        "state": _registered_state_from_disposition(value.disposition),
        "support": None if value.support is None else value.support.to_data(),
        "evidence": value.evidence,
    }


@dataclass(frozen=True, slots=True)
class ObjectSceneRegisteredWitnessCell:
    witness_id: str
    witness_digest: str
    disposition: Disposition
    support: UnitSupportInterval | None
    evidence: str
    cell_digest: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.witness_id, str)
            or _WITNESS_ID.fullmatch(self.witness_id) is None
        ):
            raise ObjectSceneVisualFrontendError(
                "registered witness ID differs"
            )
        _digest(self.witness_digest, "registered witness definition digest")
        if not isinstance(self.disposition, Disposition):
            raise TypeError("registered witness disposition differs")
        if self.support != _registered_support_from_disposition(
            self.disposition
        ):
            raise ObjectSceneVisualFrontendError(
                "registered witness support differs"
            )
        _bounded_prose(self.evidence, "registered witness evidence", 240)
        _digest(self.cell_digest, "registered witness cell digest")
        if self.cell_digest != canonical_digest(
            _registered_witness_cell_content(self)
        ):
            raise ObjectSceneVisualFrontendError(
                "registered witness cell digest differs"
            )

    @classmethod
    def create(
        cls,
        witness: ObjectSceneOperationalWitness,
        state: object,
        evidence: object,
    ) -> "ObjectSceneRegisteredWitnessCell":
        if not isinstance(witness, ObjectSceneOperationalWitness):
            raise TypeError("registered witness definition differs")
        disposition = _disposition_from_state(state)
        normalized_evidence = _normalized_bounded_prose(
            evidence, "registered witness evidence", 240
        )
        values = {
            "witness_id": witness.witness_id,
            "witness_digest": witness.witness_digest,
            "disposition": disposition,
            "support": _registered_support_from_disposition(disposition),
            "evidence": normalized_evidence,
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(
            **values,
            cell_digest=canonical_digest(
                _registered_witness_cell_content(provisional)
            ),
        )

    @classmethod
    def error(
        cls,
        witness: ObjectSceneOperationalWitness,
        evidence: object,
    ) -> "ObjectSceneRegisteredWitnessCell":
        if not isinstance(witness, ObjectSceneOperationalWitness):
            raise TypeError("registered witness definition differs")
        normalized_evidence = _normalized_bounded_prose(
            evidence, "registered witness evidence", 240
        )
        values = {
            "witness_id": witness.witness_id,
            "witness_digest": witness.witness_digest,
            "disposition": Disposition.ERROR,
            "support": None,
            "evidence": normalized_evidence,
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(
            **values,
            cell_digest=canonical_digest(
                _registered_witness_cell_content(provisional)
            ),
        )

    def to_data(self) -> dict[str, object]:
        return {
            **_registered_witness_cell_content(self),
            "cell_digest": self.cell_digest,
        }

    @classmethod
    def from_data(cls, value: object) -> "ObjectSceneRegisteredWitnessCell":
        raw = _fields(
            value,
            {
                "witness_id",
                "witness_digest",
                "state",
                "support",
                "evidence",
                "cell_digest",
            },
            "registered witness cell",
        )
        support = (
            None
            if raw["support"] is None
            else UnitSupportInterval.from_data(raw["support"])
        )
        result = cls(
            raw["witness_id"],
            raw["witness_digest"],
            _registered_disposition_from_state(raw["state"]),
            support,
            raw["evidence"],
            raw["cell_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectSceneVisualFrontendError(
                "registered witness cell is not canonical"
            )
        return result


def compile_object_scene_registered_witness_dispositions(
    values: object,
) -> Disposition:
    """Compile a conjunction without allowing uncertainty to become absence."""

    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ObjectSceneVisualFrontendError(
            "registered witness dispositions must be a finite sequence"
        )
    dispositions = tuple(values)
    if (
        not OBJECT_SCENE_MIN_REQUIRED_WITNESSES
        <= len(dispositions)
        <= OBJECT_SCENE_MAX_REQUIRED_WITNESSES
        or any(not isinstance(item, Disposition) for item in dispositions)
    ):
        raise ObjectSceneVisualFrontendError(
            "registered witness dispositions differ"
        )
    if Disposition.ERROR in dispositions:
        return Disposition.ERROR
    if Disposition.CERTIFIED_ABSENT in dispositions:
        return Disposition.CERTIFIED_ABSENT
    if all(item is Disposition.PRESENT for item in dispositions):
        return Disposition.PRESENT
    return Disposition.INDETERMINATE


def _compiled_registered_evidence(
    values: Sequence[ObjectSceneRegisteredWitnessCell],
) -> str:
    return "compiled required witness states " + "; ".join(
        f"{item.witness_id} {_registered_state_from_disposition(item.disposition)}"
        for item in values
    )


def _registered_cell_content(
    value: "ObjectSceneRegisteredTagCell",
) -> dict[str, object]:
    return {
        "tag_id": value.tag_id,
        "tag_digest": value.tag_digest,
        "witness_cells": [item.to_data() for item in value.witness_cells],
        "state": _registered_state_from_disposition(value.disposition),
        "support": None if value.support is None else value.support.to_data(),
        "evidence": value.evidence,
    }


@dataclass(frozen=True, slots=True)
class ObjectSceneRegisteredTagCell:
    tag_id: str
    tag_digest: str
    witness_cells: tuple[ObjectSceneRegisteredWitnessCell, ...]
    disposition: Disposition
    support: UnitSupportInterval | None
    evidence: str
    cell_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.tag_id, str) or _TAG_ID.fullmatch(self.tag_id) is None:
            raise ObjectSceneVisualFrontendError("registered tag ID differs")
        _digest(self.tag_digest, "registered tag definition digest")
        if (
            not isinstance(self.witness_cells, tuple)
            or not OBJECT_SCENE_MIN_REQUIRED_WITNESSES
            <= len(self.witness_cells)
            <= OBJECT_SCENE_MAX_REQUIRED_WITNESSES
            or tuple(item.witness_id for item in self.witness_cells)
            != tuple(
                f"witness_{index:02d}"
                for index in range(len(self.witness_cells))
            )
            or len({item.witness_digest for item in self.witness_cells})
            != len(self.witness_cells)
        ):
            raise ObjectSceneVisualFrontendError(
                "registered witness cells differ from frozen order"
            )
        disposition = compile_object_scene_registered_witness_dispositions(
            tuple(item.disposition for item in self.witness_cells)
        )
        if self.disposition is not disposition:
            raise ObjectSceneVisualFrontendError(
                "registered tag macro disposition differs"
            )
        if self.support != _registered_support_from_disposition(disposition):
            raise ObjectSceneVisualFrontendError("registered support differs")
        if self.evidence != _compiled_registered_evidence(self.witness_cells):
            raise ObjectSceneVisualFrontendError(
                "registered tag compiled evidence differs"
            )
        _bounded_prose(self.evidence, "registered tag evidence", 240)
        _digest(self.cell_digest, "registered tag cell digest")
        if self.cell_digest != canonical_digest(_registered_cell_content(self)):
            raise ObjectSceneVisualFrontendError("registered tag cell digest differs")

    @classmethod
    def from_witness_cells(
        cls,
        tag: "ObjectSceneSoftTag",
        witness_cells: object,
    ) -> "ObjectSceneRegisteredTagCell":
        if not isinstance(tag, ObjectSceneSoftTag):
            raise TypeError("registered tag definition differs")
        if isinstance(witness_cells, (str, bytes)) or not isinstance(
            witness_cells, Sequence
        ):
            raise ObjectSceneVisualFrontendError(
                "registered witness cell array differs"
            )
        cells = tuple(witness_cells)
        if any(
            not isinstance(item, ObjectSceneRegisteredWitnessCell)
            for item in cells
        ) or tuple(
            (item.witness_id, item.witness_digest) for item in cells
        ) != tuple(
            (item.witness_id, item.witness_digest)
            for item in tag.required_witnesses
        ):
            raise ObjectSceneVisualFrontendError(
                "registered witness cells do not exhaust frozen card"
            )
        disposition = compile_object_scene_registered_witness_dispositions(
            tuple(item.disposition for item in cells)
        )
        evidence = _compiled_registered_evidence(cells)
        values = {
            "tag_id": tag.tag_id,
            "tag_digest": tag.tag_digest,
            "witness_cells": cells,
            "disposition": disposition,
            "support": _registered_support_from_disposition(disposition),
            "evidence": evidence,
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(
            **values,
            cell_digest=canonical_digest(_registered_cell_content(provisional)),
        )

    @classmethod
    def create(
        cls,
        tag: "ObjectSceneSoftTag",
        witness_values: object,
    ) -> "ObjectSceneRegisteredTagCell":
        if not isinstance(tag, ObjectSceneSoftTag):
            raise TypeError("registered tag definition differs")
        if not isinstance(witness_values, list) or len(witness_values) != len(
            tag.required_witnesses
        ):
            raise ObjectSceneVisualFrontendError(
                "registered witness payload bounds differ"
            )
        cells: list[ObjectSceneRegisteredWitnessCell] = []
        for witness, value in zip(
            tag.required_witnesses, witness_values, strict=True
        ):
            raw = _fields(
                value,
                {"witness_id", "state", "evidence"},
                "registered witness payload",
            )
            if raw["witness_id"] != witness.witness_id:
                raise ObjectSceneVisualFrontendError(
                    "registered witness payload order differs"
                )
            try:
                cell = ObjectSceneRegisteredWitnessCell.create(
                    witness, raw["state"], raw["evidence"]
                )
            except ObjectSceneVisualFrontendError:
                cell = ObjectSceneRegisteredWitnessCell.error(
                    witness,
                    "witness cell rejected by deterministic parser",
                )
            cells.append(cell)
        return cls.from_witness_cells(tag, cells)

    def to_data(self) -> dict[str, object]:
        return {**_registered_cell_content(self), "cell_digest": self.cell_digest}

    @classmethod
    def from_data(cls, value: object) -> "ObjectSceneRegisteredTagCell":
        raw = _fields(
            value,
            {
                "tag_id",
                "tag_digest",
                "witness_cells",
                "state",
                "support",
                "evidence",
                "cell_digest",
            },
            "registered tag cell",
        )
        if not isinstance(raw["witness_cells"], list):
            raise ObjectSceneVisualFrontendError(
                "registered witness cell array differs"
            )
        support = (
            None
            if raw["support"] is None
            else UnitSupportInterval.from_data(raw["support"])
        )
        result = cls(
            raw["tag_id"],
            raw["tag_digest"],
            tuple(
                ObjectSceneRegisteredWitnessCell.from_data(item)
                for item in raw["witness_cells"]
            ),
            _registered_disposition_from_state(raw["state"]),
            support,
            raw["evidence"],
            raw["cell_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectSceneVisualFrontendError("registered tag cell is not canonical")
        return result


def _row_content(value: "ObjectSceneTranscriptObject") -> dict[str, object]:
    return {
        "object_id": value.object_id,
        "crop_receipt_digest": value.crop_receipt_digest,
        "summary": value.summary,
        "count_cells": [item.to_data() for item in value.count_cells],
        "qualitative_cells": [item.to_data() for item in value.qualitative_cells],
        "open_tags": [item.to_data() for item in value.open_tags],
        "registered_tag_cells": [item.to_data() for item in value.registered_tag_cells],
    }


@dataclass(frozen=True, slots=True)
class ObjectSceneTranscriptObject:
    object_id: str
    crop_receipt_digest: str
    summary: str
    count_cells: tuple[ObjectSceneCountCell, ...]
    qualitative_cells: tuple[ObjectSceneQualitativeCell, ...]
    open_tags: tuple[ObjectSceneOpenTag, ...]
    registered_tag_cells: tuple[ObjectSceneRegisteredTagCell, ...]
    row_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.object_id, str) or _OBJECT_ID.fullmatch(self.object_id) is None:
            raise ObjectSceneVisualFrontendError("transcript object ID differs")
        _digest(self.crop_receipt_digest, "transcript crop receipt digest")
        _bounded_prose(self.summary, "object summary", 240)
        if tuple(item.observable_id for item in self.count_cells) != OBJECT_SCENE_COUNT_OBSERVABLE_IDS:
            raise ObjectSceneVisualFrontendError("count cells do not exhaust fixed order")
        if tuple(item.observable_id for item in self.qualitative_cells) != OBJECT_SCENE_QUALITATIVE_OBSERVABLE_IDS:
            raise ObjectSceneVisualFrontendError("qualitative cells do not exhaust fixed order")
        if (
            len(self.open_tags) > OBJECT_SCENE_MAX_TAGS_PER_OBJECT
            or tuple(item.tag for item in self.open_tags)
            != tuple(sorted(set(item.tag for item in self.open_tags)))
        ):
            raise ObjectSceneVisualFrontendError("open tags differ from bounded order")
        registered_ids = tuple(item.tag_id for item in self.registered_tag_cells)
        if (
            len(registered_ids) > OBJECT_SCENE_MAX_REGISTERED_TAGS
            or registered_ids != tuple(sorted(set(registered_ids)))
        ):
            raise ObjectSceneVisualFrontendError("registered cells differ from frozen order")
        if self.open_tags and self.registered_tag_cells:
            raise ObjectSceneVisualFrontendError("transcript row mixes discovery and registered modes")
        _digest(self.row_digest, "transcript row digest")
        if self.row_digest != canonical_digest(_row_content(self)):
            raise ObjectSceneVisualFrontendError("transcript row digest differs")

    def to_data(self) -> dict[str, object]:
        return {**_row_content(self), "row_digest": self.row_digest}

    @classmethod
    def from_data(cls, value: object) -> "ObjectSceneTranscriptObject":
        raw = _fields(value, {"object_id", "crop_receipt_digest", "summary", "count_cells", "qualitative_cells", "open_tags", "registered_tag_cells", "row_digest"}, "transcript row")
        if any(not isinstance(raw[key], list) for key in ("count_cells", "qualitative_cells", "open_tags", "registered_tag_cells")):
            raise ObjectSceneVisualFrontendError("transcript row arrays differ")
        result = cls(
            raw["object_id"], raw["crop_receipt_digest"], raw["summary"],
            tuple(ObjectSceneCountCell.from_data(item) for item in raw["count_cells"]),
            tuple(ObjectSceneQualitativeCell.from_data(item) for item in raw["qualitative_cells"]),
            tuple(ObjectSceneOpenTag.from_data(item) for item in raw["open_tags"]),
            tuple(ObjectSceneRegisteredTagCell.from_data(item) for item in raw["registered_tag_cells"]),
            raw["row_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectSceneVisualFrontendError("transcript row is not canonical")
        return result


def _transcript_content(value: "ObjectSceneTranscript") -> dict[str, object]:
    return {
        "schema": OBJECT_SCENE_TRANSCRIPT_SCHEMA,
        "panel_digest": value.panel_digest,
        "inventory_digest": value.inventory_digest,
        "mode": value.mode.value,
        "registry_digest": value.registry_digest,
        "panel_summary": value.panel_summary,
        "panel_open_tags": [item.to_data() for item in value.panel_open_tags],
        "panel_registered_tag_cells": [
            item.to_data() for item in value.panel_registered_tag_cells
        ],
        "dropped_open_tags": [
            item.to_data() for item in value.dropped_open_tags
        ],
        "registered_panel_tag_ids": list(value.registered_panel_tag_ids),
        "registered_entity_tag_ids": list(value.registered_entity_tag_ids),
        "objects": [item.to_data() for item in value.objects],
        "omitted_discovery_tag_semantics": "indeterminate-never-absence",
        "dropped_discovery_tag_semantics": "indeterminate-never-absence",
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectSceneTranscript:
    panel_digest: str
    inventory_digest: str
    mode: ObjectSceneTranscriptMode
    registry_digest: str | None
    panel_summary: str
    panel_open_tags: tuple[ObjectSceneOpenTag, ...]
    panel_registered_tag_cells: tuple[ObjectSceneRegisteredTagCell, ...]
    dropped_open_tags: tuple[ObjectSceneDroppedOpenTag, ...]
    registered_panel_tag_ids: tuple[str, ...]
    registered_entity_tag_ids: tuple[str, ...]
    objects: tuple[ObjectSceneTranscriptObject, ...]
    transcript_digest: str

    def __post_init__(self) -> None:
        _digest(self.panel_digest, "transcript panel digest")
        _digest(self.inventory_digest, "transcript inventory digest")
        if not isinstance(self.mode, ObjectSceneTranscriptMode):
            raise TypeError("transcript mode differs")
        _bounded_prose(self.panel_summary, "panel summary", 360)
        if (
            len(self.panel_open_tags) > OBJECT_SCENE_MAX_TAGS_PER_PANEL
            or tuple(item.tag for item in self.panel_open_tags)
            != tuple(sorted(set(item.tag for item in self.panel_open_tags)))
        ):
            raise ObjectSceneVisualFrontendError("panel open tags differ from bounded order")
        panel_cell_ids = tuple(
            item.tag_id for item in self.panel_registered_tag_cells
        )
        if (
            len(panel_cell_ids) > OBJECT_SCENE_MAX_REGISTERED_TAGS
            or panel_cell_ids != tuple(sorted(set(panel_cell_ids)))
        ):
            raise ObjectSceneVisualFrontendError(
                "panel registered cells differ from frozen order"
            )
        if self.panel_open_tags and self.panel_registered_tag_cells:
            raise ObjectSceneVisualFrontendError(
                "transcript panel mixes discovery and registered modes"
            )
        if self.dropped_open_tags != tuple(
            sorted(self.dropped_open_tags, key=_dropped_open_tag_sort_key)
        ) or len(
            {
                (item.scope, item.owner_id, item.item_index)
                for item in self.dropped_open_tags
            }
        ) != len(self.dropped_open_tags):
            raise ObjectSceneVisualFrontendError(
                "dropped open tags differ from canonical item order"
            )
        object_ids = {item.object_id for item in self.objects}
        if any(
            (item.scope == "panel" and item.owner_id != "panel")
            or (item.scope == "entity" and item.owner_id not in object_ids)
            for item in self.dropped_open_tags
        ):
            raise ObjectSceneVisualFrontendError(
                "dropped open tag transcript owner differs"
            )
        if self.mode is ObjectSceneTranscriptMode.DISCOVERY:
            if (
                self.registry_digest is not None
                or self.panel_registered_tag_cells
                or self.registered_panel_tag_ids
                or self.registered_entity_tag_ids
                or any(item.registered_tag_cells for item in self.objects)
            ):
                raise ObjectSceneVisualFrontendError("discovery transcript carries registry cells")
        else:
            _digest(self.registry_digest, "transcript registry digest")
            if (
                self.panel_open_tags
                or self.dropped_open_tags
                or any(item.open_tags for item in self.objects)
            ):
                raise ObjectSceneVisualFrontendError("registered transcript carries open tags")
            partition_ids = (
                *self.registered_panel_tag_ids,
                *self.registered_entity_tag_ids,
            )
            if (
                any(
                    not isinstance(item, str)
                    or _TAG_ID.fullmatch(item) is None
                    for item in partition_ids
                )
                or self.registered_panel_tag_ids
                != tuple(sorted(set(self.registered_panel_tag_ids)))
                or self.registered_entity_tag_ids
                != tuple(sorted(set(self.registered_entity_tag_ids)))
                or set(self.registered_panel_tag_ids)
                & set(self.registered_entity_tag_ids)
            ):
                raise ObjectSceneVisualFrontendError(
                    "registered transcript scope partition differs"
                )
            all_ids = tuple(
                sorted(
                    (*self.registered_panel_tag_ids, *self.registered_entity_tag_ids)
                )
            )
            if all_ids != tuple(
                f"tag_{index:04d}" for index in range(len(all_ids))
            ) or len(all_ids) > OBJECT_SCENE_MAX_REGISTERED_TAGS:
                raise ObjectSceneVisualFrontendError(
                    "registered transcript catalog differs"
                )
            if panel_cell_ids != self.registered_panel_tag_ids:
                raise ObjectSceneVisualFrontendError(
                    "panel cells do not exhaust panel-scope registry"
                )
            if any(
                tuple(cell.tag_id for cell in item.registered_tag_cells)
                != self.registered_entity_tag_ids
                for item in self.objects
            ):
                raise ObjectSceneVisualFrontendError(
                    "object cells do not exhaust entity-scope registry"
                )
        if tuple(item.object_id for item in self.objects) != tuple(f"object_{index:04d}" for index in range(len(self.objects))):
            raise ObjectSceneVisualFrontendError("transcript proposal order differs")
        _digest(self.transcript_digest, "transcript digest")
        if self.transcript_digest != canonical_digest(_transcript_content(self)):
            raise ObjectSceneVisualFrontendError("transcript digest differs")

    def to_data(self) -> dict[str, object]:
        return {**_transcript_content(self), "transcript_digest": self.transcript_digest}

    @classmethod
    def from_data(cls, value: object) -> "ObjectSceneTranscript":
        expected = {
            "schema", "panel_digest", "inventory_digest", "mode",
            "registry_digest", "panel_summary", "panel_open_tags",
            "panel_registered_tag_cells", "dropped_open_tags",
            "registered_panel_tag_ids",
            "registered_entity_tag_ids", "objects",
            "omitted_discovery_tag_semantics",
            "dropped_discovery_tag_semantics", *_authority_data(),
            "transcript_digest",
        }
        raw = _fields(value, expected, "object scene transcript")
        if (
            raw["schema"] != OBJECT_SCENE_TRANSCRIPT_SCHEMA
            or raw["omitted_discovery_tag_semantics"] != "indeterminate-never-absence"
            or raw["dropped_discovery_tag_semantics"]
            != "indeterminate-never-absence"
            or any(raw[key] != item for key, item in _authority_data().items())
            or any(
                not isinstance(raw[key], list)
                for key in (
                    "panel_open_tags", "panel_registered_tag_cells",
                    "dropped_open_tags",
                    "registered_panel_tag_ids", "registered_entity_tag_ids",
                    "objects"
                )
            )
        ):
            raise ObjectSceneVisualFrontendError("transcript policy differs")
        try:
            mode = ObjectSceneTranscriptMode(raw["mode"])
        except (TypeError, ValueError) as exc:
            raise ObjectSceneVisualFrontendError("transcript mode differs") from exc
        result = cls(
            raw["panel_digest"], raw["inventory_digest"], mode,
            raw["registry_digest"], raw["panel_summary"],
            tuple(ObjectSceneOpenTag.from_data(item) for item in raw["panel_open_tags"]),
            tuple(
                ObjectSceneRegisteredTagCell.from_data(item)
                for item in raw["panel_registered_tag_cells"]
            ),
            tuple(
                ObjectSceneDroppedOpenTag.from_data(item)
                for item in raw["dropped_open_tags"]
            ),
            tuple(raw["registered_panel_tag_ids"]),
            tuple(raw["registered_entity_tag_ids"]),
            tuple(ObjectSceneTranscriptObject.from_data(item) for item in raw["objects"]),
            raw["transcript_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectSceneVisualFrontendError("transcript is not canonical")
        return result


OBJECT_SCENE_TAG_REGISTRY_SCHEMA = "gkm.object-scene-soft-tag-registry.v6"


@dataclass(frozen=True, order=True, slots=True)
class ObjectSceneSoftTag:
    tag_id: str
    scope: str
    tag: str
    distinct_panel_count: int
    orientation_constraint: str
    required_witnesses: tuple[ObjectSceneOperationalWitness, ...]
    accepted_variants: tuple[str, ...]
    near_miss_boundaries: tuple[str, ...]
    criteria_digest: str
    tag_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.tag_id, str) or _TAG_ID.fullmatch(self.tag_id) is None:
            raise ObjectSceneVisualFrontendError("soft tag ID differs")
        _soft_tag_scope(self.scope)
        _positive_tag(self.tag)
        _integer(self.distinct_panel_count, "soft tag panel count", minimum=OBJECT_SCENE_MIN_TAG_PANEL_FREQUENCY)
        _soft_tag_orientation_constraint(self.orientation_constraint)
        witnesses = canonicalize_object_scene_operational_witnesses(
            self.required_witnesses
        )
        variants = _canonical_operational_criteria(
            self.accepted_variants,
            label="accepted variants",
            minimum=0,
            maximum=OBJECT_SCENE_MAX_ACCEPTED_VARIANTS,
            normalizer=normalize_object_scene_accepted_variant,
        )
        boundaries = _canonical_operational_criteria(
            self.near_miss_boundaries,
            label="near-miss boundaries",
            minimum=0,
            maximum=OBJECT_SCENE_MAX_NEAR_MISS_BOUNDARIES,
            normalizer=normalize_object_scene_near_miss_boundary,
        )
        if (
            witnesses != self.required_witnesses
            or variants != self.accepted_variants
            or boundaries != self.near_miss_boundaries
        ):
            raise ObjectSceneVisualFrontendError(
                "soft tag operational criteria are not canonical"
            )
        _digest(self.criteria_digest, "soft tag criteria digest")
        if self.criteria_digest != object_scene_soft_tag_criteria_digest(
            self.required_witnesses,
            self.accepted_variants,
            self.near_miss_boundaries,
        ):
            raise ObjectSceneVisualFrontendError("soft tag criteria digest differs")
        _digest(self.tag_digest, "soft tag digest")
        if self.tag_digest != _soft_tag_content_digest(
            self.scope,
            self.tag,
            self.criteria_digest,
            self.orientation_constraint,
        ):
            raise ObjectSceneVisualFrontendError("soft tag content digest differs")

    @classmethod
    def create(
        cls,
        tag_id: str,
        scope: str,
        tag: str,
        distinct_panel_count: int,
        required_witnesses: object,
        accepted_variants: object = (),
        near_miss_boundaries: object = (),
        *,
        orientation_constraint: str = "bidirectional",
    ) -> "ObjectSceneSoftTag":
        normalized_scope = _soft_tag_scope(scope)
        normalized_tag = _normalized_positive_tag(tag)
        normalized_orientation_constraint = _soft_tag_orientation_constraint(
            orientation_constraint
        )
        witnesses = canonicalize_object_scene_operational_witnesses(
            required_witnesses
        )
        variants = _canonical_operational_criteria(
            accepted_variants,
            label="accepted variants",
            minimum=0,
            maximum=OBJECT_SCENE_MAX_ACCEPTED_VARIANTS,
            normalizer=normalize_object_scene_accepted_variant,
        )
        boundaries = _canonical_operational_criteria(
            near_miss_boundaries,
            label="near-miss boundaries",
            minimum=0,
            maximum=OBJECT_SCENE_MAX_NEAR_MISS_BOUNDARIES,
            normalizer=normalize_object_scene_near_miss_boundary,
        )
        criteria_digest = object_scene_soft_tag_criteria_digest(
            witnesses, variants, boundaries
        )
        return cls(
            tag_id,
            normalized_scope,
            normalized_tag,
            distinct_panel_count,
            normalized_orientation_constraint,
            witnesses,
            variants,
            boundaries,
            criteria_digest,
            _soft_tag_content_digest(
                normalized_scope,
                normalized_tag,
                criteria_digest,
                normalized_orientation_constraint,
            ),
        )

    def to_data(self) -> dict[str, object]:
        return {
            "tag_id": self.tag_id,
            "scope": self.scope,
            "tag": self.tag,
            "distinct_panel_count": self.distinct_panel_count,
            "orientation_constraint": self.orientation_constraint,
            "required_witnesses": [
                item.to_data() for item in self.required_witnesses
            ],
            "accepted_variants": list(self.accepted_variants),
            "near_miss_boundaries": list(self.near_miss_boundaries),
            "criteria_digest": self.criteria_digest,
            "tag_digest": self.tag_digest,
        }

    @classmethod
    def from_data(cls, value: object) -> "ObjectSceneSoftTag":
        raw = _fields(
            value,
            {
                "tag_id",
                "scope",
                "tag",
                "distinct_panel_count",
                "orientation_constraint",
                "required_witnesses",
                "accepted_variants",
                "near_miss_boundaries",
                "criteria_digest",
                "tag_digest",
            },
            "soft tag",
        )
        if any(
            not isinstance(raw[key], list)
            for key in (
                "required_witnesses",
                "accepted_variants",
                "near_miss_boundaries",
            )
        ):
            raise ObjectSceneVisualFrontendError("soft tag criteria arrays differ")
        result = cls(
            raw["tag_id"],
            raw["scope"],
            raw["tag"],
            raw["distinct_panel_count"],
            raw["orientation_constraint"],
            tuple(
                ObjectSceneOperationalWitness.from_data(item)
                for item in raw["required_witnesses"]
            ),
            tuple(raw["accepted_variants"]),
            tuple(raw["near_miss_boundaries"]),
            raw["criteria_digest"],
            raw["tag_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectSceneVisualFrontendError("soft tag is not canonical")
        return result


@dataclass(frozen=True, order=True, slots=True)
class ObjectSceneDroppedSoftTag:
    scope: str
    tag: str
    distinct_panel_count: int
    reason: str
    tag_digest: str

    def __post_init__(self) -> None:
        _soft_tag_scope(self.scope, "dropped soft tag scope")
        _positive_tag(self.tag)
        _integer(self.distinct_panel_count, "dropped tag panel count")
        if self.reason not in ("seen_on_fewer_than_2_panels", "registry_capacity_exceeded"):
            raise ObjectSceneVisualFrontendError("dropped tag reason differs")
        _digest(self.tag_digest, "dropped tag digest")
        if self.tag_digest != _dropped_soft_tag_content_digest(
            self.scope, self.tag
        ):
            raise ObjectSceneVisualFrontendError("dropped tag content digest differs")

    def to_data(self) -> dict[str, object]:
        return {"scope": self.scope, "tag": self.tag, "distinct_panel_count": self.distinct_panel_count, "reason": self.reason, "tag_digest": self.tag_digest}

    @classmethod
    def from_data(cls, value: object) -> "ObjectSceneDroppedSoftTag":
        raw = _fields(value, {"scope", "tag", "distinct_panel_count", "reason", "tag_digest"}, "dropped soft tag")
        result = cls(raw["scope"], raw["tag"], raw["distinct_panel_count"], raw["reason"], raw["tag_digest"])
        if result.to_data() != dict(raw):
            raise ObjectSceneVisualFrontendError("dropped tag is not canonical")
        return result


def _registry_content(value: "ObjectSceneSoftTagRegistry") -> dict[str, object]:
    return {
        "schema": OBJECT_SCENE_TAG_REGISTRY_SCHEMA,
        "source_transcript_digests": list(value.source_transcript_digests),
        "source_panel_digests": list(value.source_panel_digests),
        "minimum_distinct_panel_frequency": OBJECT_SCENE_MIN_TAG_PANEL_FREQUENCY,
        "maximum_registered_tags": OBJECT_SCENE_MAX_REGISTERED_TAGS,
        "ordering_rule": "descending-distinct-panel-frequency-then-scope-then-phrase",
        "operational_card_policy": _operational_card_policy(),
        "tags": [item.to_data() for item in value.tags],
        "dropped_tags": [item.to_data() for item in value.dropped_tags],
        "fixed_qualitative_observable_ids": list(OBJECT_SCENE_QUALITATIVE_OBSERVABLE_IDS),
        "fixed_count_observable_ids": list(OBJECT_SCENE_COUNT_OBSERVABLE_IDS),
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectSceneSoftTagRegistry:
    source_transcript_digests: tuple[str, ...]
    source_panel_digests: tuple[str, ...]
    tags: tuple[ObjectSceneSoftTag, ...]
    dropped_tags: tuple[ObjectSceneDroppedSoftTag, ...]
    registry_digest: str

    def __post_init__(self) -> None:
        if self.source_transcript_digests != tuple(sorted(set(self.source_transcript_digests))):
            raise ObjectSceneVisualFrontendError("registry transcript commitments differ")
        if self.source_panel_digests != tuple(sorted(set(self.source_panel_digests))):
            raise ObjectSceneVisualFrontendError("registry panel commitments differ")
        for item in (*self.source_transcript_digests, *self.source_panel_digests):
            _digest(item, "registry source digest")
        if len(self.tags) > OBJECT_SCENE_MAX_REGISTERED_TAGS:
            raise ObjectSceneVisualFrontendError("registered tag capacity exceeded")
        if tuple(item.tag_id for item in self.tags) != tuple(f"tag_{index:04d}" for index in range(len(self.tags))):
            raise ObjectSceneVisualFrontendError("registered tag IDs differ")
        if len({(item.scope, item.tag) for item in self.tags}) != len(self.tags):
            raise ObjectSceneVisualFrontendError(
                "registered tag inventory repeats a scoped phrase"
            )
        if tuple(
            (-item.distinct_panel_count, item.scope, item.tag)
            for item in self.tags
        ) != tuple(
            sorted(
                (-item.distinct_panel_count, item.scope, item.tag)
                for item in self.tags
            )
        ):
            raise ObjectSceneVisualFrontendError("registered tag rank differs")
        if tuple(
            (-item.distinct_panel_count, item.scope, item.tag)
            for item in self.dropped_tags
        ) != tuple(
            sorted(
                (-item.distinct_panel_count, item.scope, item.tag)
                for item in self.dropped_tags
            )
        ):
            raise ObjectSceneVisualFrontendError("dropped tag order differs")
        if len({(item.scope, item.tag) for item in self.dropped_tags}) != len(self.dropped_tags):
            raise ObjectSceneVisualFrontendError("dropped tag inventory repeats a scoped phrase")
        if (
            {(item.scope, item.tag) for item in self.tags}
            & {(item.scope, item.tag) for item in self.dropped_tags}
        ):
            raise ObjectSceneVisualFrontendError("tag is both admitted and dropped")
        _digest(self.registry_digest, "soft tag registry digest")
        if self.registry_digest != canonical_digest(_registry_content(self)):
            raise ObjectSceneVisualFrontendError("soft tag registry digest differs")

    def to_data(self) -> dict[str, object]:
        return {**_registry_content(self), "registry_digest": self.registry_digest}

    @classmethod
    def from_data(cls, value: object) -> "ObjectSceneSoftTagRegistry":
        expected = {"schema", "source_transcript_digests", "source_panel_digests", "minimum_distinct_panel_frequency", "maximum_registered_tags", "ordering_rule", "operational_card_policy", "tags", "dropped_tags", "fixed_qualitative_observable_ids", "fixed_count_observable_ids", *_authority_data(), "registry_digest"}
        raw = _fields(value, expected, "soft tag registry")
        if (
            raw["schema"] != OBJECT_SCENE_TAG_REGISTRY_SCHEMA
            or raw["minimum_distinct_panel_frequency"] != OBJECT_SCENE_MIN_TAG_PANEL_FREQUENCY
            or raw["maximum_registered_tags"] != OBJECT_SCENE_MAX_REGISTERED_TAGS
            or raw["ordering_rule"] != "descending-distinct-panel-frequency-then-scope-then-phrase"
            or raw["operational_card_policy"] != _operational_card_policy()
            or raw["fixed_qualitative_observable_ids"] != list(OBJECT_SCENE_QUALITATIVE_OBSERVABLE_IDS)
            or raw["fixed_count_observable_ids"] != list(OBJECT_SCENE_COUNT_OBSERVABLE_IDS)
            or any(raw[key] != item for key, item in _authority_data().items())
            or any(not isinstance(raw[key], list) for key in ("source_transcript_digests", "source_panel_digests", "tags", "dropped_tags"))
        ):
            raise ObjectSceneVisualFrontendError("soft tag registry policy differs")
        result = cls(tuple(raw["source_transcript_digests"]), tuple(raw["source_panel_digests"]), tuple(ObjectSceneSoftTag.from_data(item) for item in raw["tags"]), tuple(ObjectSceneDroppedSoftTag.from_data(item) for item in raw["dropped_tags"]), raw["registry_digest"])
        if result.to_data() != dict(raw):
            raise ObjectSceneVisualFrontendError("soft tag registry is not canonical")
        return result


def freeze_object_scene_soft_tag_registry(transcripts: Sequence[ObjectSceneTranscript]) -> ObjectSceneSoftTagRegistry:
    """Freeze recurring exact affirmative tags without seeing any labels."""

    values = tuple(transcripts)
    if any(not isinstance(item, ObjectSceneTranscript) or item.mode is not ObjectSceneTranscriptMode.DISCOVERY for item in values):
        raise ObjectSceneVisualFrontendError("registry inputs must be discovery transcripts")
    if len({item.transcript_digest for item in values}) != len(values):
        raise ObjectSceneVisualFrontendError("registry repeats a discovery transcript")
    panels_by_tag: dict[tuple[str, str], set[str]] = {}
    for transcript in values:
        for observed in transcript.panel_open_tags:
            panels_by_tag.setdefault(("panel", observed.tag), set()).add(
                transcript.panel_digest
            )
        for row in transcript.objects:
            for observed in row.open_tags:
                panels_by_tag.setdefault(("entity", observed.tag), set()).add(
                    transcript.panel_digest
                )
    ranked = sorted(
        (
            (scope, tag, len(panels))
            for (scope, tag), panels in panels_by_tag.items()
        ),
        key=lambda item: (-item[2], item[0], item[1]),
    )
    eligible = [
        item
        for item in ranked
        if item[2] >= OBJECT_SCENE_MIN_TAG_PANEL_FREQUENCY
    ]
    admitted = eligible[:OBJECT_SCENE_MAX_REGISTERED_TAGS]
    drops: list[ObjectSceneDroppedSoftTag] = []
    for scope, tag, count in ranked:
        if count < OBJECT_SCENE_MIN_TAG_PANEL_FREQUENCY:
            reason = "seen_on_fewer_than_2_panels"
        elif (scope, tag, count) not in admitted:
            reason = "registry_capacity_exceeded"
        else:
            continue
        drops.append(
            ObjectSceneDroppedSoftTag(
                scope,
                tag,
                count,
                reason,
                _dropped_soft_tag_content_digest(scope, tag),
            )
        )
    tags = tuple(
        ObjectSceneSoftTag.create(
            f"tag_{index:04d}",
            scope,
            tag,
            count,
            ({"kind": "shape_appearance", "statement": tag},),
            (),
            (),
            orientation_constraint="bidirectional",
        )
        for index, (scope, tag, count) in enumerate(admitted)
    )
    drops_tuple = tuple(
        sorted(
            drops,
            key=lambda item: (
                -item.distinct_panel_count,
                item.scope,
                item.tag,
            ),
        )
    )
    values_map = {
        "source_transcript_digests": tuple(sorted(item.transcript_digest for item in values)),
        "source_panel_digests": tuple(sorted(set(item.panel_digest for item in values))),
        "tags": tags,
        "dropped_tags": drops_tuple,
    }
    provisional = object.__new__(ObjectSceneSoftTagRegistry)
    for name, item in values_map.items():
        object.__setattr__(provisional, name, item)
    return ObjectSceneSoftTagRegistry(**values_map, registry_digest=canonical_digest(_registry_content(provisional)))


def verify_object_scene_soft_tag_registry(
    registry: ObjectSceneSoftTagRegistry,
    transcripts: Sequence[ObjectSceneTranscript],
    *,
    expected_registry_digest: str | None = None,
) -> ObjectSceneSoftTagRegistry:
    if not isinstance(registry, ObjectSceneSoftTagRegistry):
        raise TypeError("registry must be ObjectSceneSoftTagRegistry")
    restored = ObjectSceneSoftTagRegistry.from_data(registry.to_data())
    if expected_registry_digest is not None and restored.registry_digest != _digest(
        expected_registry_digest, "expected registry digest"
    ):
        raise ObjectSceneVisualFrontendError("soft tag registry differs from commitment")
    if freeze_object_scene_soft_tag_registry(transcripts) != restored:
        raise ObjectSceneVisualFrontendError("soft tag registry differs from discovery replay")
    return restored


def _verify_registered_cell_card_binding(
    cell: ObjectSceneRegisteredTagCell,
    tag: ObjectSceneSoftTag,
) -> None:
    if (
        not isinstance(cell, ObjectSceneRegisteredTagCell)
        or not isinstance(tag, ObjectSceneSoftTag)
        or cell.tag_id != tag.tag_id
        or cell.tag_digest != tag.tag_digest
        or tuple(
            (item.witness_id, item.witness_digest)
            for item in cell.witness_cells
        )
        != tuple(
            (item.witness_id, item.witness_digest)
            for item in tag.required_witnesses
        )
    ):
        raise ObjectSceneVisualFrontendError(
            "registered cell card binding differs"
        )


_QUALITATIVE_MEANINGS = {
    "triangle_like": "has the visible overall form of a triangle",
    "quadrilateral_like": "has the visible overall form of a quadrilateral outline",
    "sector_like": "resembles a wedge or circular sector",
    "bird_like": "resembles a bird or flying bird silhouette",
    "open_contour": "has a contour with visible open ends",
    "closed_boundary": "has at least one closed enclosing boundary",
    "pointed": "has a conspicuous pointed tip",
    "thin_elongated": "is conspicuously thin and elongated",
    "necked": "has a narrow neck between wider portions",
    "mismatched_parts": "contains visibly dissimilar joined or paired portions",
    "unequal_part_sizes": "contains corresponding portions of visibly unequal size",
    "unequal_edge_lengths": "has boundary edges of visibly unequal length",
    "reflection_symmetry": "has visible approximate reflection symmetry",
    "bilateral_layout": "has two visually corresponding portions in opposite regions",
    "oblique": "contains a prominent direction tilted from horizontal and vertical",
    "parallel": "contains at least two approximately parallel straight directions",
    "perpendicular": "contains approximately perpendicular straight directions",
    "crossing": "contains visible strokes or boundaries that cross",
    "internal_marks": "contains marks visibly inside an enclosing boundary",
    "paired_sector_mismatch": "contains a paired sector-like form whose corresponding portions visibly mismatch",
    "triangle_with_three_lines": "contains a triangle-like form with three distinct internal line marks",
}

_COUNT_MEANINGS = {
    "straight_segment_count": "visible straight line segments",
    "curved_segment_count": "visible curved line segments",
    "endpoint_count": "visible open stroke or contour endpoints",
    "junction_count": "visible junctions where lines or boundaries meet",
    "closed_loop_count": "visible closed loops or enclosing boundaries",
    "internal_mark_count": "distinct visible marks inside enclosing boundaries",
    "connected_part_count": "visibly connected constituent portions",
    "acute_angle_count": "visible angles smaller than a right angle",
    "obtuse_angle_count": "visible angles larger than a right angle and smaller than a straight angle",
    "right_angle_count": "visible approximately right angles",
}


def object_scene_transcript_output_schema(
    inventory: ObjectSceneProposalInventory,
    mode: ObjectSceneTranscriptMode,
    registry: ObjectSceneSoftTagRegistry | None = None,
) -> dict[str, object]:
    if not isinstance(inventory, ObjectSceneProposalInventory):
        raise TypeError("inventory must be ObjectSceneProposalInventory")
    if not isinstance(mode, ObjectSceneTranscriptMode):
        raise TypeError("mode must be ObjectSceneTranscriptMode")
    if (mode is ObjectSceneTranscriptMode.REGISTERED_EVALUATION) != isinstance(registry, ObjectSceneSoftTagRegistry):
        raise ObjectSceneVisualFrontendError("transcript mode/registry differ")
    nullable_integer = {"anyOf": [{"type": "integer"}, {"type": "null"}]}
    count_cell = {
        "type": "object",
        "properties": {
            "observable_id": {"type": "string", "enum": list(OBJECT_SCENE_COUNT_OBSERVABLE_IDS)},
            "state": {"type": "string", "enum": ["measured", "indeterminate"]},
            "lower_count": nullable_integer,
            "upper_count": nullable_integer,
            "evidence": {"type": "string"},
        },
        "required": ["observable_id", "state", "lower_count", "upper_count", "evidence"],
        "additionalProperties": False,
    }
    qualitative_cell = {
        "type": "object",
        "properties": {
            "observable_id": {"type": "string", "enum": list(OBJECT_SCENE_QUALITATIVE_OBSERVABLE_IDS)},
            "state": {"type": "string", "enum": ["present", "absent", "indeterminate"]},
            "evidence": {"type": "string"},
        },
        "required": ["observable_id", "state", "evidence"],
        "additionalProperties": False,
    }
    open_tag = {
        "type": "object",
        "properties": {
            "tag": {
                "type": "string",
                "description": (
                    "Lowercase affirmative visual phrase; at most "
                    f"{OBJECT_SCENE_MAX_TAG_CHARACTERS} characters."
                ),
            },
            "state": {"type": "string", "enum": ["present", "indeterminate"]},
            "evidence": {"type": "string"},
        },
        "required": ["tag", "state", "evidence"],
        "additionalProperties": False,
    }
    def registered_cell(scope: str) -> dict[str, object]:
        registered_id_schema: dict[str, object] = {"type": "string"}
        if registry is not None:
            scoped_ids = [
                item.tag_id for item in registry.tags if item.scope == scope
            ]
            if scoped_ids:
                registered_id_schema["enum"] = scoped_ids
        return {
            "type": "object",
            "properties": {
                "tag_id": registered_id_schema,
                "witness_cells": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "witness_id": {"type": "string"},
                            "state": {
                                "type": "string",
                                "enum": [
                                    "present",
                                    "absent",
                                    "indeterminate",
                                ],
                            },
                            "evidence": {"type": "string"},
                        },
                        "required": ["witness_id", "state", "evidence"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["tag_id", "witness_cells"],
            "additionalProperties": False,
        }
    row = {
        "type": "object",
        "properties": {
            "object_id": {"type": "string"},
            "summary": {"type": "string"},
            "counts": {"type": "array", "items": count_cell},
            "observables": {"type": "array", "items": qualitative_cell},
            "open_tags": {
                "type": "array",
                "description": (
                    "At most "
                    f"{OBJECT_SCENE_MAX_TAGS_PER_OBJECT} optional visual phrases."
                ),
                "items": open_tag,
            },
            "registered_tags": {
                "type": "array",
                "items": registered_cell("entity"),
            },
        },
        "required": ["object_id", "summary", "counts", "observables", "open_tags", "registered_tags"],
        "additionalProperties": False,
    }
    panel_row = {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "open_tags": {
                "type": "array",
                "description": (
                    "At most "
                    f"{OBJECT_SCENE_MAX_TAGS_PER_PANEL} optional visual phrases."
                ),
                "items": open_tag,
            },
            "registered_tags": {
                "type": "array",
                "items": registered_cell("panel"),
            },
        },
        "required": ["summary", "open_tags", "registered_tags"],
        "additionalProperties": False,
    }
    schema = {
        "type": "object",
        "properties": {
            "panel": panel_row,
            "objects": {"type": "array", "items": row},
        },
        "required": ["panel", "objects"],
        "additionalProperties": False,
    }
    validate_codex_strict_output_schema(schema)
    return schema


def _registered_operational_card_block(
    registry: ObjectSceneSoftTagRegistry,
    scope: str,
) -> str:
    binding = {
        "panel": "the whole panel",
        "entity": "one frozen proposal",
    }[_soft_tag_scope(scope)]
    rows: list[str] = []
    for item in registry.tags:
        if item.scope != scope:
            continue
        witnesses = "\n".join(
            f"    - {witness.witness_id} [{witness.kind}]: "
            f"{witness.statement}"
            for witness in item.required_witnesses
        )
        variants = "\n".join(
            f"    - {variant}" for variant in item.accepted_variants
        ) or "    - none declared"
        boundaries = "\n".join(
            f"    - {boundary}" for boundary in item.near_miss_boundaries
        ) or "    - none declared"
        rows.append(
            f"- {item.tag_id}\n"
            f"  phrase: {item.tag}\n"
            f"  binding: {binding}\n"
            f"  required witnesses:\n{witnesses}\n"
            f"  accepted variants:\n{variants}\n"
            f"  near-miss boundaries:\n{boundaries}"
        )
    return "\n".join(rows) or "- no registered cards"


def object_scene_transcript_prompt(
    inventory: ObjectSceneProposalInventory,
    mode: ObjectSceneTranscriptMode,
    registry: ObjectSceneSoftTagRegistry | None = None,
) -> str:
    object_scene_transcript_output_schema(inventory, mode, registry)
    mapping = "\n".join(
        f"- {item.object_id}: {item.atlas_name}, row {item.row_index}, column {item.column_index}"
        for item in inventory.objects
    ) or "- no frozen proposals"
    count_lines = "\n".join(f"- {key}: {value}" for key, value in _COUNT_MEANINGS.items())
    qualitative_lines = "\n".join(f"- {key}: {value}" for key, value in _QUALITATIVE_MEANINGS.items())
    if mode is ObjectSceneTranscriptMode.DISCOVERY:
        mode_text = (
            "The whole-panel open_tags may contain at most eight short atomic "
            "affirmative visual phrases about the complete composition, "
            "including visible relations or arrangements spanning proposals. "
            "Each proposal's open_tags may contain at most eight short atomic "
            "affirmative visual phrases about that proposal itself. "
            f"Every open_tags phrase must contain at most {OBJECT_SCENE_MAX_TAG_CHARACTERS} characters. "
            "Natural internal visible relations using 'and' or 'than' are allowed, "
            "but negation, alternatives, and experimental vocabulary are forbidden. "
            "Use normalized lowercase phrases in lexical order. Each may be present or indeterminate, never absent. "
            "Omission means only unrecorded and remains indeterminate; omission never means absent. "
            "All registered_tags arrays must be empty."
        )
    else:
        assert registry is not None
        panel_tag_lines = _registered_operational_card_block(registry, "panel")
        entity_tag_lines = _registered_operational_card_block(registry, "entity")
        mode_text = (
            "All open_tags arrays must be empty. The whole-panel registered_tags "
            "must contain every whole-panel registered tag, and only those "
            "tags, in the exact order below. Every proposal's registered_tags "
            "must contain every per-proposal registered tag, and only those "
            "tags, in the exact order below. "
            "For each tag, return every required witness as a separate "
            "witness_cell in the exact listed witness order. Judge each "
            "witness only on the tag's stated binding. Return present when "
            "that witness is visibly supported, absent only when it is "
            "visibly contradicted, and indeterminate otherwise. Apply the "
            "accepted variants and near-miss boundaries while interpreting "
            "every witness, but neither guidance list is itself a witness and "
            "neither casts a vote. Do not return a combined tag state; it is "
            "computed later from the witness cells. Give every witness concise "
            "visible evidence.\n"
            f"Whole-panel registered cards:\n{panel_tag_lines}\n"
            f"Per-proposal registered cards:\n{entity_tag_lines}"
        )
    prompt = (
        "You are a neutral empirical visual observer. Inspect panel.png for complete context and every objects_NNN.png atlas for detail. "
        "Each atlas is a four by four row-major array. Empty slots are irrelevant. Use the exact atlas map below. "
        "Describe only visible appearance. Do not infer hidden identities or experimental intent. "
        "Return one panel record describing the complete visible composition, "
        "plus exactly one row for every frozen proposal in object ID order. "
        "The panel record is not a semantic object and must describe relations "
        "or arrangements spanning proposals when visible. Include a concise "
        "neutral summary and concise visible evidence in every cell. "
        "For every count, return the narrowest defensible integer interval from zero through 999; use measured with both bounds, or indeterminate with both bounds null. "
        "For every qualitative observable, return present, absent, or indeterminate. Exhaust both fixed lists in their shown order.\n\n"
        f"Atlas map:\n{mapping}\n\nCount meanings:\n{count_lines}\n\nQualitative meanings:\n{qualitative_lines}\n\n{mode_text}"
    )
    envelope = prompt + "\n" + json.dumps(object_scene_transcript_output_schema(inventory, mode, registry), sort_keys=True) + "\npanel.png\n" + "\n".join(item.name for item in inventory.atlas_sheets)
    if _FORBIDDEN_VISIBLE.search(envelope) is not None:
        raise ObjectSceneVisualFrontendError("model-visible transcript envelope contains experimental vocabulary")
    _scene_runtime._assert_model_visible_boundary(
        prompt,
        object_scene_transcript_output_schema(inventory, mode, registry),
        ("panel.png", *(item.name for item in inventory.atlas_sheets)),
        hidden_values=(inventory.panel_digest, inventory.inventory_digest, *( () if registry is None else (registry.registry_digest,) )),
        allowed_visual_words=("side", "path"),
    )
    return prompt


@dataclass(frozen=True, slots=True)
class ObjectScenePreparedTranscriptInputs:
    inventory_digest: str
    mode: ObjectSceneTranscriptMode
    registry_digest: str | None
    prompt: str
    output_schema: Mapping[str, Any]
    presentation: tuple[tuple[str, bytes], ...]
    presentation_identities: tuple[PrototypeImageIdentity, ...]
    preparation_digest: str


def _presentation_identities(presentation: Sequence[tuple[str, bytes]]) -> tuple[PrototypeImageIdentity, ...]:
    return tuple(PrototypeImageIdentity(name, len(data), hashlib.sha256(data).hexdigest()) for name, data in presentation)


def prepare_object_scene_transcript_inputs(
    png_bytes: bytes,
    inventory: ObjectSceneProposalInventory,
    mode: ObjectSceneTranscriptMode,
    registry: ObjectSceneSoftTagRegistry | None = None,
) -> ObjectScenePreparedTranscriptInputs:
    verify_object_scene_proposal_inventory(inventory, png_bytes)
    atlas = render_object_scene_proposal_atlas(inventory, png_bytes)
    presentation = (("panel.png", png_bytes), *atlas)
    prompt = object_scene_transcript_prompt(inventory, mode, registry)
    schema = object_scene_transcript_output_schema(inventory, mode, registry)
    identities = _presentation_identities(presentation)
    digest = canonical_digest({
        "schema": "gkm.object-scene-prepared-transcript.v6",
        "inventory_digest": inventory.inventory_digest,
        "mode": mode.value,
        "registry_digest": None if registry is None else registry.registry_digest,
        "prompt_digest": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "output_schema_digest": canonical_digest(schema),
        "presentation": [item.to_data() for item in identities],
    })
    return ObjectScenePreparedTranscriptInputs(inventory.inventory_digest, mode, None if registry is None else registry.registry_digest, prompt, schema, presentation, identities, digest)


def _parse_transcript_tag_payload(
    open_values: object,
    registered_values: object,
    *,
    mode: ObjectSceneTranscriptMode,
    registry_tags: tuple[ObjectSceneSoftTag, ...],
    maximum_open_tags: int,
    label: str,
    scope: str,
    owner_id: str,
) -> tuple[
    tuple[ObjectSceneOpenTag, ...],
    tuple[ObjectSceneRegisteredTagCell, ...],
    tuple[ObjectSceneDroppedOpenTag, ...],
]:
    if not isinstance(open_values, list) or not isinstance(registered_values, list):
        raise ObjectSceneVisualFrontendError(f"{label} tag arrays differ")
    open_tags: list[ObjectSceneOpenTag] = []
    registered_cells: list[ObjectSceneRegisteredTagCell] = []
    dropped: list[ObjectSceneDroppedOpenTag] = []
    if mode is ObjectSceneTranscriptMode.DISCOVERY:
        if registered_values:
            raise ObjectSceneVisualFrontendError(f"{label} discovery tag bounds differ")
        candidates: list[tuple[ObjectSceneOpenTag, int, object]] = []

        def drop(index: int, value: object, reason: str) -> None:
            dropped.append(
                ObjectSceneDroppedOpenTag.create(
                    scope=scope,
                    owner_id=owner_id,
                    item_index=index,
                    raw_item=value,
                    reason_code=reason,
                )
            )

        for index, value in enumerate(open_values):
            try:
                tag = _fields(
                    value,
                    {"tag", "state", "evidence"},
                    f"{label} open tag payload",
                )
            except ObjectSceneVisualFrontendError:
                drop(index, value, "malformed_item")
                continue
            try:
                phrase = _normalized_positive_tag(tag["tag"])
            except ObjectSceneVisualFrontendError:
                drop(index, value, "invalid_tag")
                continue
            if tag["state"] not in ("present", "indeterminate"):
                drop(index, value, "invalid_state")
                continue
            try:
                evidence = _normalized_bounded_prose(
                    tag["evidence"], "open tag evidence", 240
                )
            except ObjectSceneVisualFrontendError:
                drop(index, value, "invalid_evidence")
                continue
            candidates.append(
                (
                    ObjectSceneOpenTag.create(
                        phrase, tag["state"], evidence
                    ),
                    index,
                    value,
                )
            )

        grouped: dict[str, list[tuple[ObjectSceneOpenTag, int, object]]] = {}
        for candidate in candidates:
            grouped.setdefault(candidate[0].tag, []).append(candidate)
        unique: list[tuple[ObjectSceneOpenTag, int, object]] = []
        for phrase in sorted(grouped):
            group = grouped[phrase]
            if len(group) == 1:
                unique.append(group[0])
                continue
            for _, index, value in group:
                drop(index, value, "duplicate_normalized_tag")
        for observed, _, _ in unique[:maximum_open_tags]:
            open_tags.append(observed)
        for _, index, value in unique[maximum_open_tags:]:
            drop(index, value, "capacity_exceeded")
        dropped.sort(key=_dropped_open_tag_sort_key)
    else:
        if open_values or len(registered_values) != len(registry_tags):
            raise ObjectSceneVisualFrontendError(
                f"{label} registered tag bounds differ"
            )
        for expected_tag, value in zip(
            registry_tags, registered_values, strict=True
        ):
            tag = _fields(
                value, {"tag_id", "witness_cells"},
                f"{label} registered tag payload",
            )
            if tag["tag_id"] != expected_tag.tag_id:
                raise ObjectSceneVisualFrontendError(
                    f"{label} registered tag payload order differs"
                )
            registered_cells.append(
                ObjectSceneRegisteredTagCell.create(
                    expected_tag, tag["witness_cells"]
                )
            )
    return tuple(open_tags), tuple(registered_cells), tuple(dropped)


def _parse_object_scene_transcript_payload(
    payload: Mapping[str, Any],
    inventory: ObjectSceneProposalInventory,
    mode: ObjectSceneTranscriptMode,
    registry: ObjectSceneSoftTagRegistry | None,
) -> ObjectSceneTranscript:
    canonical = _canonical_payload(payload)
    raw = _fields(canonical, {"panel", "objects"}, "transcript payload")
    if not isinstance(raw["objects"], list) or len(raw["objects"]) != len(inventory.objects):
        raise ObjectSceneVisualFrontendError("payload does not exhaust frozen proposals")
    rows: list[ObjectSceneTranscriptObject] = []
    panel_registry_tags = (
        ()
        if registry is None
        else tuple(
            item for item in registry.tags if item.scope == "panel"
        )
    )
    entity_registry_tags = (
        ()
        if registry is None
        else tuple(
            item for item in registry.tags if item.scope == "entity"
        )
    )
    panel_registry_ids = tuple(item.tag_id for item in panel_registry_tags)
    entity_registry_ids = tuple(item.tag_id for item in entity_registry_tags)
    panel = _fields(
        raw["panel"], {"summary", "open_tags", "registered_tags"},
        "payload panel",
    )
    panel_open_tags, panel_registered_cells, panel_dropped = (
        _parse_transcript_tag_payload(
            panel["open_tags"],
            panel["registered_tags"],
            mode=mode,
            registry_tags=panel_registry_tags,
            maximum_open_tags=OBJECT_SCENE_MAX_TAGS_PER_PANEL,
            label="panel",
            scope="panel",
            owner_id="panel",
        )
    )
    dropped_open_tags = list(panel_dropped)
    for index, (item, crop) in enumerate(zip(raw["objects"], inventory.objects, strict=True)):
        row = _fields(item, {"object_id", "summary", "counts", "observables", "open_tags", "registered_tags"}, f"payload object {index}")
        if row["object_id"] != crop.object_id or any(not isinstance(row[key], list) for key in ("counts", "observables", "open_tags", "registered_tags")):
            raise ObjectSceneVisualFrontendError("payload object binding differs")
        if len(row["counts"]) != len(OBJECT_SCENE_COUNT_OBSERVABLE_IDS) or len(row["observables"]) != len(OBJECT_SCENE_QUALITATIVE_OBSERVABLE_IDS):
            raise ObjectSceneVisualFrontendError("payload fixed cells are incomplete")
        counts: list[ObjectSceneCountCell] = []
        for expected_id, value in zip(OBJECT_SCENE_COUNT_OBSERVABLE_IDS, row["counts"], strict=True):
            cell = _fields(value, {"observable_id", "state", "lower_count", "upper_count", "evidence"}, "count payload cell")
            if cell["observable_id"] != expected_id:
                raise ObjectSceneVisualFrontendError("count payload order differs")
            counts.append(ObjectSceneCountCell.create(expected_id, cell["state"], cell["lower_count"], cell["upper_count"], cell["evidence"]))
        qualities: list[ObjectSceneQualitativeCell] = []
        for expected_id, value in zip(OBJECT_SCENE_QUALITATIVE_OBSERVABLE_IDS, row["observables"], strict=True):
            cell = _fields(value, {"observable_id", "state", "evidence"}, "qualitative payload cell")
            if cell["observable_id"] != expected_id:
                raise ObjectSceneVisualFrontendError("qualitative payload order differs")
            qualities.append(ObjectSceneQualitativeCell.create(expected_id, cell["state"], cell["evidence"]))
        open_tags, registered_cells, row_dropped = _parse_transcript_tag_payload(
            row["open_tags"], row["registered_tags"], mode=mode,
            registry_tags=entity_registry_tags,
            maximum_open_tags=OBJECT_SCENE_MAX_TAGS_PER_OBJECT,
            label=f"object {index}",
            scope="entity",
            owner_id=crop.object_id,
        )
        dropped_open_tags.extend(row_dropped)
        row_values = {
            "object_id": crop.object_id,
            "crop_receipt_digest": crop.receipt_digest,
            "summary": _normalized_bounded_prose(row["summary"], "object summary", 240),
            "count_cells": tuple(counts),
            "qualitative_cells": tuple(qualities),
            "open_tags": open_tags,
            "registered_tag_cells": registered_cells,
        }
        provisional = object.__new__(ObjectSceneTranscriptObject)
        for name, value in row_values.items():
            object.__setattr__(provisional, name, value)
        rows.append(ObjectSceneTranscriptObject(**row_values, row_digest=canonical_digest(_row_content(provisional))))
    transcript_values = {
        "panel_digest": inventory.panel_digest,
        "inventory_digest": inventory.inventory_digest,
        "mode": mode,
        "registry_digest": None if registry is None else registry.registry_digest,
        "panel_summary": _normalized_bounded_prose(
            panel["summary"], "panel summary", 360
        ),
        "panel_open_tags": panel_open_tags,
        "panel_registered_tag_cells": panel_registered_cells,
        "dropped_open_tags": tuple(
            sorted(dropped_open_tags, key=_dropped_open_tag_sort_key)
        ),
        "registered_panel_tag_ids": panel_registry_ids,
        "registered_entity_tag_ids": entity_registry_ids,
        "objects": tuple(rows),
    }
    provisional_transcript = object.__new__(ObjectSceneTranscript)
    for name, value in transcript_values.items():
        object.__setattr__(provisional_transcript, name, value)
    return ObjectSceneTranscript(**transcript_values, transcript_digest=canonical_digest(_transcript_content(provisional_transcript)))


def object_scene_transcript_protocol_digest() -> str:
    return canonical_digest({
        "schema": "gkm.object-scene-transcript-protocol.v6",
        "frontend_id": OBJECT_SCENE_FRONTEND_ID,
        "source_digest": object_scene_visual_frontend_source_digest(),
        "inventory_protocol_digest": object_scene_inventory_protocol_digest(),
        "transport_source_digest": _scene_runtime.prototype_scene_transport_source_digest(),
        "presentation": "opaque-panel-plus-row-major-proposal-atlas",
        "physical_calls_per_artifact": 1,
        "fixed_count_observable_ids": list(OBJECT_SCENE_COUNT_OBSERVABLE_IDS),
        "fixed_qualitative_observable_ids": list(OBJECT_SCENE_QUALITATIVE_OBSERVABLE_IDS),
        "discovery_tag_cap_per_proposal": OBJECT_SCENE_MAX_TAGS_PER_OBJECT,
        "discovery_tag_cap_for_whole_panel": OBJECT_SCENE_MAX_TAGS_PER_PANEL,
        "discovery_tag_max_characters": OBJECT_SCENE_MAX_TAG_CHARACTERS,
        "discovery_tag_drop_reason_codes": list(
            OBJECT_SCENE_DISCOVERY_OPEN_TAG_DROP_REASON_CODES
        ),
        "discovery_tag_drop_record_schema": (
            OBJECT_SCENE_DROPPED_OPEN_TAG_SCHEMA
        ),
        "discovery_tag_item_failure_semantics": (
            "quarantine-item-preserve-mandatory-transcript"
        ),
        "discovery_tag_drop_binding": (
            "scope-owner-id-item-index-exact-raw-item-digest"
        ),
        "registered_operational_card_policy": _operational_card_policy(),
        "registered_operational_card_digest_chain": (
            "criteria-digest-then-scope-phrase-orientation-criteria-tag-digest-"
            "then-registry"
        ),
        "registered_operational_card_model_view": (
            "tag-id-phrase-scope-derived-binding-ordered-witness-id-kind-statement-"
            "accepted-variants-near-miss-boundaries"
        ),
        "registered_model_output": (
            "tag-id-plus-ordered-witness-id-state-evidence-no-macro-state"
        ),
        "registered_macro_compiler": (
            "error-dominant-then-any-absent-then-all-present-else-indeterminate"
        ),
        "whole_panel_summary_and_tag_cells": True,
        "whole_panel_is_not_a_semantic_object": True,
        "model_visible_unicode_normalization": (
            "predeclared-ordinary-punctuation-to-canonical-ascii-before-validation"
        ),
        "discovery_open_tag_input_order": (
            "unordered-normalize-drop-all-duplicate-members-then-lexical-cap"
        ),
        "registry_minimum_distinct_panel_frequency": OBJECT_SCENE_MIN_TAG_PANEL_FREQUENCY,
        "registry_capacity": OBJECT_SCENE_MAX_REGISTERED_TAGS,
        "registry_tag_scopes": list(OBJECT_SCENE_SOFT_TAG_SCOPES),
        "registry_identity": (
            "scope-plus-normalized-affirmative-phrase-plus-criteria-digest-"
            "plus-orientation-constraint"
        ),
        "registry_ordering": (
            "descending-distinct-panel-frequency-then-scope-then-phrase"
        ),
        "registered_scope_partition": (
            "whole-panel-cells-exhaust-panel-scope-and-each-proposal-exhausts-entity-scope"
        ),
        "omitted_or_unregistered_tag": "indeterminate-never-absence",
        "failure_semantics": (
            "mandatory-and-registered-errors-fail-closed-never-absence;"
            "optional-discovery-tag-item-errors-are-quarantined"
        ),
        "repeated_registered_calls": "same-visible-envelope-distinct-opaque-context",
        **_authority_data(),
    })


def _artifact_content(value: "ObjectSceneTranscriptArtifact") -> dict[str, object]:
    return {
        "schema": OBJECT_SCENE_TRANSCRIPT_ARTIFACT_SCHEMA,
        "scene_id": value.scene_id,
        "observation_context_digest": value.observation_context_digest,
        "panel_digest": value.panel_digest,
        "inventory": value.inventory.to_data(),
        "inventory_digest": value.inventory_digest,
        "mode": value.mode.value,
        "registry": None if value.registry is None else value.registry.to_data(),
        "registry_digest": value.registry_digest,
        "source_digest": value.source_digest,
        "inventory_protocol_digest": value.inventory_protocol_digest,
        "transcript_protocol_digest": value.transcript_protocol_digest,
        "preparation_digest": value.preparation_digest,
        "prompt_digest": value.prompt_digest,
        "output_schema_digest": value.output_schema_digest,
        "model": value.model,
        "reasoning_effort": value.reasoning_effort,
        "model_request_digest": value.model_request_digest,
        "expected_launcher_digest": value.expected_launcher_digest,
        "cloud_policy_cache_binding": value.cloud_policy_cache_binding,
        "model_catalog_digest": value.model_catalog_digest,
        "no_tools_attestation_digest": value.no_tools_attestation_digest,
        "presentation": [item.to_data() for item in value.presentation],
        "physical_call_count": value.physical_call_count,
        "status": value.status.value,
        "model_payload": value.model_payload,
        "payload_freeze_digest": value.payload_freeze_digest,
        "receipt": None if value.receipt is None else value.receipt.to_dict(),
        "transcript": None if value.transcript is None else value.transcript.to_data(),
        "failure_code": value.failure_code,
        "failure_type": value.failure_type,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectSceneTranscriptArtifact:
    scene_id: str
    observation_context_digest: str
    panel_digest: str
    inventory: ObjectSceneProposalInventory
    inventory_digest: str
    mode: ObjectSceneTranscriptMode
    registry: ObjectSceneSoftTagRegistry | None
    registry_digest: str | None
    source_digest: str
    inventory_protocol_digest: str
    transcript_protocol_digest: str
    preparation_digest: str
    prompt_digest: str
    output_schema_digest: str
    model: str
    reasoning_effort: str
    model_request_digest: str
    expected_launcher_digest: str
    cloud_policy_cache_binding: str
    model_catalog_digest: str
    no_tools_attestation_digest: str
    presentation: tuple[PrototypeImageIdentity, ...]
    physical_call_count: int
    status: PrototypeSceneObserverStatus
    model_payload: Mapping[str, Any] | None
    payload_freeze_digest: str | None
    receipt: CodexReceipt | None
    transcript: ObjectSceneTranscript | None
    failure_code: str | None
    failure_type: str | None
    artifact_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.scene_id, str) or not 1 <= len(self.scene_id) <= 512 or _PROSE.fullmatch(self.scene_id) is None:
            raise ObjectSceneVisualFrontendError("scene ID differs")
        _address(self.observation_context_digest, "observation context digest")
        for name in (
            "panel_digest", "inventory_digest", "source_digest", "inventory_protocol_digest",
            "transcript_protocol_digest", "preparation_digest", "prompt_digest",
            "output_schema_digest", "model_request_digest", "expected_launcher_digest",
            "model_catalog_digest", "no_tools_attestation_digest", "artifact_digest",
        ):
            _digest(getattr(self, name), name)
        if not isinstance(self.inventory, ObjectSceneProposalInventory) or self.inventory.inventory_digest != self.inventory_digest or self.inventory.panel_digest != self.panel_digest:
            raise ObjectSceneVisualFrontendError("artifact inventory binding differs")
        if self.source_digest != object_scene_visual_frontend_source_digest() or self.inventory_protocol_digest != object_scene_inventory_protocol_digest() or self.transcript_protocol_digest != object_scene_transcript_protocol_digest():
            raise ObjectSceneVisualFrontendError("artifact protocol binding differs")
        if not isinstance(self.mode, ObjectSceneTranscriptMode):
            raise TypeError("artifact mode differs")
        if self.mode is ObjectSceneTranscriptMode.DISCOVERY:
            if self.registry is not None or self.registry_digest is not None:
                raise ObjectSceneVisualFrontendError("discovery artifact carries registry")
        elif not isinstance(self.registry, ObjectSceneSoftTagRegistry) or self.registry.registry_digest != self.registry_digest:
            raise ObjectSceneVisualFrontendError("registered artifact binding differs")
        if not isinstance(self.model, str) or not self.model or not isinstance(self.reasoning_effort, str) or not self.reasoning_effort:
            raise ObjectSceneVisualFrontendError("artifact model request differs")
        if self.model_request_digest != _scene_runtime.prototype_scene_observer_model_digest(self.model, self.reasoning_effort):
            raise ObjectSceneVisualFrontendError("artifact model digest differs")
        if self.cloud_policy_cache_binding != "absent":
            _address(self.cloud_policy_cache_binding, "cloud policy cache binding")
        if not isinstance(self.presentation, tuple) or not self.presentation or self.presentation[0].name != "panel.png":
            raise ObjectSceneVisualFrontendError("artifact presentation differs")
        if self.physical_call_count != 1:
            raise ObjectSceneVisualFrontendError("artifact physical call count differs")
        if not isinstance(self.status, PrototypeSceneObserverStatus) or self.status not in (
            PrototypeSceneObserverStatus.SUCCESS,
            PrototypeSceneObserverStatus.PARSER_ERROR,
            PrototypeSceneObserverStatus.TRANSPORT_ERROR,
        ):
            raise ObjectSceneVisualFrontendError("artifact status differs")
        if self.status is PrototypeSceneObserverStatus.SUCCESS:
            if self.model_payload is None or self.receipt is None or not isinstance(self.transcript, ObjectSceneTranscript) or self.failure_code is not None or self.failure_type is not None:
                raise ObjectSceneVisualFrontendError("successful transcript artifact differs")
        elif self.status is PrototypeSceneObserverStatus.PARSER_ERROR:
            if self.model_payload is None or self.receipt is None or self.transcript is not None or self.failure_code != "payload_rejected" or self.failure_type != "ObjectSceneTranscriptPayloadError":
                raise ObjectSceneVisualFrontendError("parser-error transcript artifact differs")
        else:
            if self.model_payload is not None or self.receipt is not None or self.transcript is not None or self.failure_code != "transport_failed" or not isinstance(self.failure_type, str) or _CODE.fullmatch(self.failure_type) is None:
                raise ObjectSceneVisualFrontendError("transport-error transcript artifact differs")
        if self.model_payload is None:
            if self.payload_freeze_digest is not None:
                raise ObjectSceneVisualFrontendError("unreceipted artifact has payload digest")
        else:
            canonical = _canonical_payload(self.model_payload)
            if dict(self.model_payload) != canonical or self.payload_freeze_digest != canonical_digest(canonical):
                raise ObjectSceneVisualFrontendError("artifact payload freeze differs")
        if self.transcript is not None and (
            self.transcript.panel_digest != self.panel_digest
            or self.transcript.inventory_digest != self.inventory_digest
            or self.transcript.mode is not self.mode
            or self.transcript.registry_digest != self.registry_digest
        ):
            raise ObjectSceneVisualFrontendError("artifact transcript parent differs")
        if self.transcript is not None and self.registry is not None:
            expected_panel_ids = tuple(
                item.tag_id
                for item in self.registry.tags
                if item.scope == "panel"
            )
            expected_entity_ids = tuple(
                item.tag_id
                for item in self.registry.tags
                if item.scope == "entity"
            )
            if (
                self.transcript.registered_panel_tag_ids
                != expected_panel_ids
                or self.transcript.registered_entity_tag_ids
                != expected_entity_ids
            ):
                raise ObjectSceneVisualFrontendError(
                    "artifact transcript registry scope partition differs"
                )
            tags_by_id = {item.tag_id: item for item in self.registry.tags}
            transcript_cells = (
                *self.transcript.panel_registered_tag_cells,
                *(
                    cell
                    for row in self.transcript.objects
                    for cell in row.registered_tag_cells
                ),
            )
            for cell in transcript_cells:
                tag = tags_by_id.get(cell.tag_id)
                if tag is None:
                    raise ObjectSceneVisualFrontendError(
                        "artifact registered cell card binding differs"
                    )
                try:
                    _verify_registered_cell_card_binding(cell, tag)
                except ObjectSceneVisualFrontendError as exc:
                    raise ObjectSceneVisualFrontendError(
                        "artifact registered cell card binding differs"
                    ) from exc
        _digest(self.artifact_digest, "object scene artifact digest")
        if self.artifact_digest != canonical_digest(_artifact_content(self)):
            raise ObjectSceneVisualFrontendError("object scene artifact digest differs")

    def to_data(self) -> dict[str, object]:
        return {**_artifact_content(self), "artifact_digest": self.artifact_digest}

    @classmethod
    def from_data(cls, value: object, *, expected_artifact_digest: str | None = None) -> "ObjectSceneTranscriptArtifact":
        expected = {"schema", "scene_id", "observation_context_digest", "panel_digest", "inventory", "inventory_digest", "mode", "registry", "registry_digest", "source_digest", "inventory_protocol_digest", "transcript_protocol_digest", "preparation_digest", "prompt_digest", "output_schema_digest", "model", "reasoning_effort", "model_request_digest", "expected_launcher_digest", "cloud_policy_cache_binding", "model_catalog_digest", "no_tools_attestation_digest", "presentation", "physical_call_count", "status", "model_payload", "payload_freeze_digest", "receipt", "transcript", "failure_code", "failure_type", *_authority_data(), "artifact_digest"}
        raw = _fields(value, expected, "object scene transcript artifact")
        if raw["schema"] != OBJECT_SCENE_TRANSCRIPT_ARTIFACT_SCHEMA or any(raw[key] != item for key, item in _authority_data().items()) or not isinstance(raw["presentation"], list):
            raise ObjectSceneVisualFrontendError("object scene artifact policy differs")
        try:
            mode = ObjectSceneTranscriptMode(raw["mode"])
            status = PrototypeSceneObserverStatus(raw["status"])
        except (TypeError, ValueError) as exc:
            raise ObjectSceneVisualFrontendError("object scene artifact enum differs") from exc
        registry = None if raw["registry"] is None else ObjectSceneSoftTagRegistry.from_data(raw["registry"])
        receipt = None if raw["receipt"] is None else _scene_runtime._receipt_from_data(raw["receipt"])
        transcript = None if raw["transcript"] is None else ObjectSceneTranscript.from_data(raw["transcript"])
        payload = None if raw["model_payload"] is None else _canonical_payload(raw["model_payload"])
        result = cls(
            raw["scene_id"], raw["observation_context_digest"], raw["panel_digest"],
            ObjectSceneProposalInventory.from_data(raw["inventory"]), raw["inventory_digest"],
            mode, registry, raw["registry_digest"], raw["source_digest"],
            raw["inventory_protocol_digest"], raw["transcript_protocol_digest"],
            raw["preparation_digest"], raw["prompt_digest"], raw["output_schema_digest"],
            raw["model"], raw["reasoning_effort"], raw["model_request_digest"],
            raw["expected_launcher_digest"], raw["cloud_policy_cache_binding"],
            raw["model_catalog_digest"], raw["no_tools_attestation_digest"],
            tuple(PrototypeImageIdentity.from_data(item) for item in raw["presentation"]),
            raw["physical_call_count"], status, payload, raw["payload_freeze_digest"],
            receipt, transcript, raw["failure_code"], raw["failure_type"], raw["artifact_digest"],
        )
        if expected_artifact_digest is not None and result.artifact_digest != _digest(expected_artifact_digest, "expected artifact digest"):
            raise ObjectSceneVisualFrontendError("object scene artifact differs from commitment")
        if result.to_data() != dict(raw):
            raise ObjectSceneVisualFrontendError("object scene artifact is not canonical")
        return result

    def assert_untampered(self) -> None:
        ObjectSceneTranscriptArtifact.from_data(self.to_data(), expected_artifact_digest=self.artifact_digest)


def _build_transcript_artifact(
    *,
    scene_id: str,
    observation_context_digest: str,
    inventory: ObjectSceneProposalInventory,
    mode: ObjectSceneTranscriptMode,
    registry: ObjectSceneSoftTagRegistry | None,
    prepared: ObjectScenePreparedTranscriptInputs,
    model: str,
    reasoning_effort: str,
    expected_launcher_digest: str,
    cloud_policy_cache_binding: str,
    model_catalog_digest: str,
    no_tools_attestation_digest: str,
    status: PrototypeSceneObserverStatus,
    payload: Mapping[str, Any] | None,
    receipt: CodexReceipt | None,
    transcript: ObjectSceneTranscript | None,
    failure_code: str | None,
    failure_type: str | None,
) -> ObjectSceneTranscriptArtifact:
    canonical_payload = None if payload is None else _canonical_payload(payload)
    values = {
        "scene_id": scene_id,
        "observation_context_digest": _address(observation_context_digest, "observation context digest"),
        "panel_digest": inventory.panel_digest,
        "inventory": inventory,
        "inventory_digest": inventory.inventory_digest,
        "mode": mode,
        "registry": registry,
        "registry_digest": None if registry is None else registry.registry_digest,
        "source_digest": object_scene_visual_frontend_source_digest(),
        "inventory_protocol_digest": object_scene_inventory_protocol_digest(),
        "transcript_protocol_digest": object_scene_transcript_protocol_digest(),
        "preparation_digest": prepared.preparation_digest,
        "prompt_digest": hashlib.sha256(prepared.prompt.encode("utf-8")).hexdigest(),
        "output_schema_digest": canonical_digest(prepared.output_schema),
        "model": model,
        "reasoning_effort": reasoning_effort,
        "model_request_digest": _scene_runtime.prototype_scene_observer_model_digest(model, reasoning_effort),
        "expected_launcher_digest": expected_launcher_digest,
        "cloud_policy_cache_binding": cloud_policy_cache_binding,
        "model_catalog_digest": model_catalog_digest,
        "no_tools_attestation_digest": no_tools_attestation_digest,
        "presentation": prepared.presentation_identities,
        "physical_call_count": 1,
        "status": status,
        "model_payload": canonical_payload,
        "payload_freeze_digest": None if canonical_payload is None else canonical_digest(canonical_payload),
        "receipt": receipt,
        "transcript": transcript,
        "failure_code": failure_code,
        "failure_type": failure_type,
    }
    provisional = object.__new__(ObjectSceneTranscriptArtifact)
    for name, item in values.items():
        object.__setattr__(provisional, name, item)
    return ObjectSceneTranscriptArtifact(**values, artifact_digest=canonical_digest(_artifact_content(provisional)))


def observe_object_scene_transcript(
    exact_png_bytes: bytes,
    *,
    scene_id: str,
    observation_context_digest: str,
    mode: ObjectSceneTranscriptMode,
    registry: ObjectSceneSoftTagRegistry | None = None,
    inventory: ObjectSceneProposalInventory | None = None,
    expected_panel_sha256: str | None = None,
    model: str,
    reasoning_effort: str = "medium",
    minutes: int = 15,
    verbose: bool = False,
    executable: str = "codex",
    cloud_policy_cache_snapshot: CloudPolicyCacheSnapshot | None = None,
    expected_launcher_digest: str,
    model_catalog_snapshot: CodexModelCatalogSnapshot,
    no_tools_attestation: CodexNoToolsAttestation,
    transport: Any = run_codex_named_images_structured,
) -> ObjectSceneTranscriptArtifact:
    """Make exactly one neutral visual call over one frozen proposal inventory."""

    if not isinstance(exact_png_bytes, bytes):
        raise TypeError("exact PNG input must be bytes")
    if not callable(transport):
        raise TypeError("transport must be callable")
    if expected_panel_sha256 is not None and hashlib.sha256(exact_png_bytes).hexdigest() != _digest(
        expected_panel_sha256, "expected panel SHA-256"
    ):
        raise ObjectSceneVisualFrontendError("exact PNG differs from external commitment")
    frozen = extract_object_scene_proposal_inventory(exact_png_bytes) if inventory is None else verify_object_scene_proposal_inventory(inventory, exact_png_bytes)
    prepared = prepare_object_scene_transcript_inputs(exact_png_bytes, frozen, mode, registry)
    policy_binding = _scene_runtime._policy_cache_binding(cloud_policy_cache_snapshot)
    model_catalog_digest, no_tools_attestation_digest = _scene_runtime._validate_no_tools_runtime(
        model_catalog_snapshot=model_catalog_snapshot,
        no_tools_attestation=no_tools_attestation,
        expected_launcher_digest=expected_launcher_digest,
        cloud_policy_cache_binding=policy_binding,
    )
    try:
        payload, receipt = _scene_runtime._stage_and_call(
            prepared.presentation,
            prompt=prepared.prompt,
            schema=prepared.output_schema,
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
        failure_type = type(exc).__name__
        if _CODE.fullmatch(failure_type) is None:
            failure_type = "UnclassifiedTransportFailure"
        return _build_transcript_artifact(
            scene_id=scene_id, observation_context_digest=observation_context_digest,
            inventory=frozen, mode=mode, registry=registry, prepared=prepared,
            model=model, reasoning_effort=reasoning_effort,
            expected_launcher_digest=expected_launcher_digest,
            cloud_policy_cache_binding=policy_binding,
            model_catalog_digest=model_catalog_digest,
            no_tools_attestation_digest=no_tools_attestation_digest,
            status=PrototypeSceneObserverStatus.TRANSPORT_ERROR,
            payload=None, receipt=None, transcript=None,
            failure_code="transport_failed", failure_type=failure_type,
        )
    try:
        transcript = _parse_object_scene_transcript_payload(payload, frozen, mode, registry)
    except (ObjectSceneVisualFrontendError, TypeError, ValueError):
        return _build_transcript_artifact(
            scene_id=scene_id, observation_context_digest=observation_context_digest,
            inventory=frozen, mode=mode, registry=registry, prepared=prepared,
            model=model, reasoning_effort=reasoning_effort,
            expected_launcher_digest=expected_launcher_digest,
            cloud_policy_cache_binding=policy_binding,
            model_catalog_digest=model_catalog_digest,
            no_tools_attestation_digest=no_tools_attestation_digest,
            status=PrototypeSceneObserverStatus.PARSER_ERROR,
            payload=payload, receipt=receipt, transcript=None,
            failure_code="payload_rejected", failure_type="ObjectSceneTranscriptPayloadError",
        )
    return _build_transcript_artifact(
        scene_id=scene_id, observation_context_digest=observation_context_digest,
        inventory=frozen, mode=mode, registry=registry, prepared=prepared,
        model=model, reasoning_effort=reasoning_effort,
        expected_launcher_digest=expected_launcher_digest,
        cloud_policy_cache_binding=policy_binding,
        model_catalog_digest=model_catalog_digest,
        no_tools_attestation_digest=no_tools_attestation_digest,
        status=PrototypeSceneObserverStatus.SUCCESS,
        payload=payload, receipt=receipt, transcript=transcript,
        failure_code=None, failure_type=None,
    )


def verify_object_scene_transcript_artifact(
    artifact: ObjectSceneTranscriptArtifact,
    exact_png_bytes: bytes,
    *,
    expected_scene_id: str,
    expected_observation_context_digest: str,
    expected_panel_sha256: str,
    expected_artifact_digest: str,
) -> ObjectSceneTranscriptArtifact:
    """Cold replay pixels, atlas, envelope, receipt, payload, and transcript."""

    if not isinstance(artifact, ObjectSceneTranscriptArtifact):
        raise TypeError("artifact must be ObjectSceneTranscriptArtifact")
    artifact.assert_untampered()
    if artifact.artifact_digest != _digest(expected_artifact_digest, "expected artifact digest"):
        raise ObjectSceneVisualFrontendError("artifact differs from commitment")
    if artifact.scene_id != expected_scene_id or artifact.observation_context_digest != _address(expected_observation_context_digest, "expected observation context digest"):
        raise ObjectSceneVisualFrontendError("artifact external context differs")
    panel_digest = hashlib.sha256(exact_png_bytes).hexdigest()
    if panel_digest != _digest(expected_panel_sha256, "expected panel SHA-256") or panel_digest != artifact.panel_digest:
        raise ObjectSceneVisualFrontendError("artifact panel bytes differ")
    rebuilt = verify_object_scene_proposal_inventory(artifact.inventory, exact_png_bytes, expected_inventory_digest=artifact.inventory_digest)
    prepared = prepare_object_scene_transcript_inputs(exact_png_bytes, rebuilt, artifact.mode, artifact.registry)
    if (
        prepared.preparation_digest != artifact.preparation_digest
        or hashlib.sha256(prepared.prompt.encode("utf-8")).hexdigest() != artifact.prompt_digest
        or canonical_digest(prepared.output_schema) != artifact.output_schema_digest
        or prepared.presentation_identities != artifact.presentation
    ):
        raise ObjectSceneVisualFrontendError("artifact visible envelope differs from replay")
    if artifact.receipt is not None:
        assert artifact.model_payload is not None
        with tempfile.TemporaryDirectory(prefix="bongard-object-scene-replay-") as raw:
            directory = Path(raw)
            paths: list[str] = []
            names: list[str] = []
            for name, data in prepared.presentation:
                target = directory / name
                target.write_bytes(data)
                paths.append(str(target.resolve()))
                names.append(name)
            try:
                validate_codex_named_image_receipt(
                    artifact.receipt, prepared.prompt, tuple(paths), tuple(names),
                    prepared.output_schema, artifact.model_payload,
                )
            except Exception as exc:
                raise ObjectSceneVisualFrontendError("artifact receipt replay failed") from exc
        if (
            artifact.receipt.requested_model != artifact.model
            or artifact.receipt.requested_reasoning_effort != artifact.reasoning_effort
            or artifact.receipt.codex_launcher_digest != artifact.expected_launcher_digest
            or artifact.receipt.model_catalog_digest != artifact.model_catalog_digest
            or artifact.receipt.tool_surface_attestation_digest != artifact.no_tools_attestation_digest
        ):
            raise ObjectSceneVisualFrontendError("artifact receipt runtime binding differs")
    if artifact.status is PrototypeSceneObserverStatus.SUCCESS:
        assert artifact.model_payload is not None and artifact.transcript is not None
        replayed = _parse_object_scene_transcript_payload(artifact.model_payload, rebuilt, artifact.mode, artifact.registry)
        if replayed != artifact.transcript:
            raise ObjectSceneVisualFrontendError("artifact transcript replay differs")
    elif artifact.status is PrototypeSceneObserverStatus.PARSER_ERROR:
        assert artifact.model_payload is not None
        try:
            _parse_object_scene_transcript_payload(artifact.model_payload, rebuilt, artifact.mode, artifact.registry)
        except (ObjectSceneVisualFrontendError, TypeError, ValueError):
            pass
        else:
            raise ObjectSceneVisualFrontendError("parser-error artifact now parses")
    if ObjectSceneTranscriptArtifact.from_data(artifact.to_data(), expected_artifact_digest=expected_artifact_digest) != artifact:
        raise ObjectSceneVisualFrontendError("artifact cold round trip differs")
    return artifact


@dataclass(frozen=True, slots=True)
class ObjectSceneSoftTagLookup:
    tag: str
    tag_id: str | None
    disposition: Disposition
    support: UnitSupportInterval | None
    evidence: str


def lookup_object_scene_soft_tag(
    source: ObjectSceneTranscript | ObjectSceneTranscriptArtifact,
    object_id: str,
    tag: str,
    *,
    registry: ObjectSceneSoftTagRegistry | None = None,
) -> ObjectSceneSoftTagLookup:
    """Resolve a tag without ever converting omission or failure to absence."""

    phrase = _positive_tag(tag)
    if isinstance(source, ObjectSceneTranscriptArtifact):
        if source.transcript is None:
            return ObjectSceneSoftTagLookup(phrase, None, Disposition.ERROR, None, "visual transcript unavailable")
        transcript = source.transcript
        frozen_registry = source.registry
    elif isinstance(source, ObjectSceneTranscript):
        transcript = source
        frozen_registry = registry
    else:
        raise TypeError("soft tag source differs")
    rows = {item.object_id: item for item in transcript.objects}
    if object_id not in rows:
        raise ObjectSceneVisualFrontendError("soft tag object ID differs")
    row = rows[object_id]
    if transcript.mode is ObjectSceneTranscriptMode.DISCOVERY:
        for item in row.open_tags:
            if item.tag == phrase:
                return ObjectSceneSoftTagLookup(phrase, None, item.disposition, item.support, item.evidence)
        return ObjectSceneSoftTagLookup(phrase, None, Disposition.INDETERMINATE, UnitSupportInterval(0, 1), "affirmative phrase was not recorded in bounded discovery")
    if frozen_registry is None or frozen_registry.registry_digest != transcript.registry_digest:
        raise ObjectSceneVisualFrontendError("soft tag lookup registry differs")
    by_tag = {
        item.tag: item
        for item in frozen_registry.tags
        if item.scope == "entity"
    }
    registered = by_tag.get(phrase)
    if registered is None:
        return ObjectSceneSoftTagLookup(phrase, None, Disposition.INDETERMINATE, UnitSupportInterval(0, 1), "affirmative phrase is outside the frozen registry")
    cells_by_id = {item.tag_id: item for item in row.registered_tag_cells}
    cell = cells_by_id.get(registered.tag_id)
    if cell is None:
        raise ObjectSceneVisualFrontendError(
            "registered entity cell is missing from transcript"
        )
    _verify_registered_cell_card_binding(cell, registered)
    return ObjectSceneSoftTagLookup(phrase, registered.tag_id, cell.disposition, cell.support, cell.evidence)


def lookup_object_scene_panel_soft_tag(
    source: ObjectSceneTranscript | ObjectSceneTranscriptArtifact,
    tag: str,
    *,
    registry: ObjectSceneSoftTagRegistry | None = None,
) -> ObjectSceneSoftTagLookup:
    """Resolve a whole-panel tag with omission/failure kept non-negative."""

    phrase = _positive_tag(tag)
    if isinstance(source, ObjectSceneTranscriptArtifact):
        if source.transcript is None:
            return ObjectSceneSoftTagLookup(
                phrase,
                None,
                Disposition.ERROR,
                None,
                "visual transcript unavailable",
            )
        transcript = source.transcript
        frozen_registry = source.registry
    elif isinstance(source, ObjectSceneTranscript):
        transcript = source
        frozen_registry = registry
    else:
        raise TypeError("soft tag source differs")
    if transcript.mode is ObjectSceneTranscriptMode.DISCOVERY:
        for item in transcript.panel_open_tags:
            if item.tag == phrase:
                return ObjectSceneSoftTagLookup(
                    phrase,
                    None,
                    item.disposition,
                    item.support,
                    item.evidence,
                )
        return ObjectSceneSoftTagLookup(
            phrase,
            None,
            Disposition.INDETERMINATE,
            UnitSupportInterval(0, 1),
            "affirmative phrase was not recorded in bounded panel discovery",
        )
    if (
        frozen_registry is None
        or frozen_registry.registry_digest != transcript.registry_digest
    ):
        raise ObjectSceneVisualFrontendError("soft tag lookup registry differs")
    registered = next(
        (
            item
            for item in frozen_registry.tags
            if item.scope == "panel" and item.tag == phrase
        ),
        None,
    )
    if registered is None:
        return ObjectSceneSoftTagLookup(
            phrase,
            None,
            Disposition.INDETERMINATE,
            UnitSupportInterval(0, 1),
            "affirmative phrase is outside the frozen panel registry",
        )
    cells_by_id = {
        item.tag_id: item for item in transcript.panel_registered_tag_cells
    }
    cell = cells_by_id.get(registered.tag_id)
    if cell is None:
        raise ObjectSceneVisualFrontendError(
            "registered panel cell is missing from transcript"
        )
    _verify_registered_cell_card_binding(cell, registered)
    return ObjectSceneSoftTagLookup(
        phrase,
        registered.tag_id,
        cell.disposition,
        cell.support,
        cell.evidence,
    )


__all__ = [
    "CountInterval",
    "OBJECT_SCENE_COUNT_OBSERVABLE_IDS",
    "OBJECT_SCENE_DISCOVERY_OPEN_TAG_DROP_REASON_CODES",
    "OBJECT_SCENE_DROPPED_OPEN_TAG_SCHEMA",
    "OBJECT_SCENE_SOFT_TAG_ORIENTATION_CONSTRAINTS",
    "OBJECT_SCENE_MAX_ACCEPTED_VARIANTS",
    "OBJECT_SCENE_MAX_CRITERION_CHARACTERS",
    "OBJECT_SCENE_MAX_NEAR_MISS_BOUNDARIES",
    "OBJECT_SCENE_MAX_REGISTERED_TAGS",
    "OBJECT_SCENE_MAX_REQUIRED_WITNESSES",
    "OBJECT_SCENE_MAX_TAG_CHARACTERS",
    "OBJECT_SCENE_MAX_TAGS_PER_PANEL",
    "OBJECT_SCENE_MAX_TAGS_PER_OBJECT",
    "OBJECT_SCENE_MIN_CRITERION_CHARACTERS",
    "OBJECT_SCENE_MIN_REQUIRED_WITNESSES",
    "OBJECT_SCENE_OPERATIONAL_WITNESS_KINDS",
    "OBJECT_SCENE_QUALITATIVE_OBSERVABLE_IDS",
    "OBJECT_SCENE_SOFT_TAG_SCOPES",
    "ObjectSceneCountCell",
    "ObjectSceneCropReceipt",
    "ObjectSceneDroppedOpenTag",
    "ObjectSceneDroppedSoftTag",
    "ObjectSceneOpenTag",
    "ObjectSceneOperationalWitness",
    "ObjectScenePreparedTranscriptInputs",
    "ObjectSceneProposalInventory",
    "ObjectSceneQualitativeCell",
    "ObjectSceneRegisteredTagCell",
    "ObjectSceneRegisteredWitnessCell",
    "ObjectSceneSoftTag",
    "ObjectSceneSoftTagLookup",
    "ObjectSceneSoftTagRegistry",
    "ObjectSceneTranscript",
    "ObjectSceneTranscriptArtifact",
    "ObjectSceneTranscriptMode",
    "ObjectSceneTranscriptObject",
    "ObjectSceneVisualFrontendError",
    "UnitSupportInterval",
    "extract_object_scene_proposal_inventory",
    "canonicalize_object_scene_operational_witnesses",
    "compile_object_scene_registered_witness_dispositions",
    "freeze_object_scene_soft_tag_registry",
    "lookup_object_scene_panel_soft_tag",
    "lookup_object_scene_soft_tag",
    "normalize_object_scene_accepted_variant",
    "normalize_object_scene_near_miss_boundary",
    "normalize_object_scene_witness_statement",
    "object_scene_inventory_protocol_digest",
    "object_scene_soft_tag_criteria_digest",
    "object_scene_transcript_output_schema",
    "object_scene_transcript_prompt",
    "object_scene_transcript_protocol_digest",
    "object_scene_visual_frontend_source_digest",
    "observe_object_scene_transcript",
    "prepare_object_scene_transcript_inputs",
    "render_object_scene_proposal_atlas",
    "verify_object_scene_proposal_inventory",
    "verify_object_scene_soft_tag_registry",
    "verify_object_scene_transcript_artifact",
]

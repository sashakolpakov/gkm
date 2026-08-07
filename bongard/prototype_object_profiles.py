"""Canonical closed object-profile language and pure-Python evaluator.

The records in this module consume already-produced integer observations.  They
never inspect pixels, invoke a model, execute candidate supplied text, or claim
that a connected-component grouping is a semantic object.  A hypothesis is
only a candidate-independent binding of low-level components and masks.
"""

from __future__ import annotations

from bongard.runtime_source_snapshot import capture_loaded_source


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)

from dataclasses import dataclass
from enum import Enum
import re
from typing import Any, Mapping, Sequence

from bongard.canonical import canonical_digest
from bongard.evidence import Disposition
from bongard.python_predicate_authority import PYTHON_PREDICATE_AUTHORITY_ID


OBJECT_FEATURE_CATALOG_SCHEMA = "gkm.bongard-object-feature-catalog.v1"
OBJECT_PROFILE_SCHEMA = "gkm.bongard-object-profile.v1"
OBJECT_LOCAL_OBSERVATION_PACKET_SCHEMA = (
    "gkm.bongard-object-local-observation-packet.v1"
)
OBJECT_PROFILE_EVALUATION_SCHEMA = "gkm.bongard-object-profile-evaluation.v1"
PPM_SCALE = 1_000_000

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_CODE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")


class ObjectProfileError(ValueError):
    """A closed profile, observation packet, or digest is malformed."""


class ObjectProfileOperator(str, Enum):
    EQUALS = "equals"
    AT_LEAST = "at_least"


def _authority_data() -> dict[str, object]:
    return {
        "predicate_authority_id": PYTHON_PREDICATE_AUTHORITY_ID,
        "python_is_canonical_authority": True,
        "lean_present": False,
        "lean_required": False,
        "lean_defines_identity_or_decision": False,
        "lean_required_for_replay": False,
        "lean_removal_changes_decision": False,
    }


def _exact_fields(value: object, expected: set[str], label: str) -> Mapping[str, Any]:
    if (
        not isinstance(value, Mapping)
        or any(not isinstance(key, str) for key in value)
        or set(value) != expected
    ):
        raise ObjectProfileError(f"{label} fields differ from the closed schema")
    return value


def _code(value: object, label: str) -> str:
    if not isinstance(value, str) or _CODE.fullmatch(value) is None:
        raise ObjectProfileError(f"{label} must be a bounded code")
    return value


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ObjectProfileError(f"{label} must be a lowercase sha256")
    return value


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    if value < minimum:
        raise ObjectProfileError(f"{label} must be at least {minimum}")
    return value


def _optional_text(value: object, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value or value != value.strip():
        raise ObjectProfileError(f"{label} must be null or nonempty stripped text")
    return value


@dataclass(frozen=True, order=True, slots=True)
class ObjectFeatureSpec:
    feature_id: str
    unit: str
    maximum: int | None
    operational_description: str
    allowed_comparators: tuple[ObjectProfileOperator, ...]

    def __post_init__(self) -> None:
        _code(self.feature_id, "feature_id")
        if self.unit not in {"count", "ppm"}:
            raise ObjectProfileError("feature unit must be count or ppm")
        if self.unit == "ppm":
            if self.maximum != PPM_SCALE:
                raise ObjectProfileError("ppm features must have maximum 1000000")
            expected = (ObjectProfileOperator.AT_LEAST,)
        elif self.maximum is not None:
            raise ObjectProfileError("count features must be unbounded above")
        else:
            expected = (
                ObjectProfileOperator.EQUALS,
                ObjectProfileOperator.AT_LEAST,
            )
        _optional_text(self.operational_description, "operational_description")
        if self.allowed_comparators != expected:
            raise ObjectProfileError("feature comparators differ from the closed grammar")

    def to_data(self) -> dict[str, object]:
        return {
            "feature_id": self.feature_id,
            "unit": self.unit,
            "minimum": 0,
            "maximum": self.maximum,
            "operational_description": self.operational_description,
            "allowed_comparators": [item.value for item in self.allowed_comparators],
        }


_COUNT_COMPARATORS = (
    ObjectProfileOperator.EQUALS,
    ObjectProfileOperator.AT_LEAST,
)
_PPM_COMPARATORS = (ObjectProfileOperator.AT_LEAST,)
OBJECT_FEATURE_CATALOG = (
    ObjectFeatureSpec("straight_span_count", "count", None, "Detected approximately straight spans.", _COUNT_COMPARATORS),
    ObjectFeatureSpec("inward_arc_count", "count", None, "Detected arcs curving toward the bound mask interior.", _COUNT_COMPARATORS),
    ObjectFeatureSpec("open_outline_support_ppm", "ppm", PPM_SCALE, "Support for an outline with a detected opening, in parts per million.", _PPM_COMPARATORS),
    ObjectFeatureSpec("endpoint_count", "count", None, "Detected skeleton or stroke endpoints.", _COUNT_COMPARATORS),
    ObjectFeatureSpec("branch_count", "count", None, "Detected skeleton or stroke branch points.", _COUNT_COMPARATORS),
    ObjectFeatureSpec("cycle_count", "count", None, "Detected independent closed cycles.", _COUNT_COMPARATORS),
    ObjectFeatureSpec("pointed_terminal_appendage_count", "count", None, "Detected terminal appendages ending in a pointed tip.", _COUNT_COMPARATORS),
    ObjectFeatureSpec("oblique_span_support_ppm", "ppm", PPM_SCALE, "Support for an obliquely oriented span, in parts per million.", _PPM_COMPARATORS),
    ObjectFeatureSpec("rounded_leaf_support_ppm", "ppm", PPM_SCALE, "Support for a rounded leaf-like contour arrangement, in parts per million.", _PPM_COMPARATORS),
    ObjectFeatureSpec("bird_like_support_ppm", "ppm", PPM_SCALE, "Support for a bird-like contour arrangement, in parts per million.", _PPM_COMPARATORS),
    ObjectFeatureSpec("paired_sector_mismatch_support_ppm", "ppm", PPM_SCALE, "Support for a mismatched pair of sector-like subshapes, in parts per million.", _PPM_COMPARATORS),
    ObjectFeatureSpec("triangle_with_three_lines_support_ppm", "ppm", PPM_SCALE, "Support for a triangular subshape accompanied by three line-like spans, in parts per million.", _PPM_COMPARATORS),
    ObjectFeatureSpec("sector_like_support_ppm", "ppm", PPM_SCALE, "Support for a sector-like contour arrangement, in parts per million.", _PPM_COMPARATORS),
    ObjectFeatureSpec("triangle_subshape_count", "count", None, "Detected triangular subshapes.", _COUNT_COMPARATORS),
    ObjectFeatureSpec("additional_straight_line_count", "count", None, "Detected straight line-like spans outside detected triangular subshapes.", _COUNT_COMPARATORS),
)
OBJECT_FEATURE_IDS = tuple(item.feature_id for item in OBJECT_FEATURE_CATALOG)
_FEATURE_BY_ID = {item.feature_id: item for item in OBJECT_FEATURE_CATALOG}


def object_feature_catalog_data() -> dict[str, object]:
    return {
        "schema": OBJECT_FEATURE_CATALOG_SCHEMA,
        "features": [item.to_data() for item in OBJECT_FEATURE_CATALOG],
        "exhaustive": True,
        "order_is_semantic": True,
        "source_digest": _LOADED_SOURCE_SHA256,
        **_authority_data(),
    }


def object_feature_catalog_digest() -> str:
    return canonical_digest(object_feature_catalog_data())


OBJECT_FEATURE_CATALOG_DIGEST = object_feature_catalog_digest()


@dataclass(frozen=True, order=True, slots=True)
class IntegerInterval:
    lower: int
    upper: int

    def __post_init__(self) -> None:
        _integer(self.lower, "interval lower")
        _integer(self.upper, "interval upper")
        if self.lower > self.upper:
            raise ObjectProfileError("interval lower exceeds upper")

    def to_data(self) -> dict[str, int]:
        return {"lower": self.lower, "upper": self.upper}

    @classmethod
    def from_data(cls, value: object) -> "IntegerInterval":
        raw = _exact_fields(value, {"lower", "upper"}, "integer interval")
        return cls(raw["lower"], raw["upper"])


class ObjectFeatureCellState(str, Enum):
    SCORED = "scored"
    INDETERMINATE = "indeterminate"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class ObjectFeatureCell:
    hypothesis_id: str
    feature_id: str
    state: ObjectFeatureCellState
    interval: IntegerInterval | None
    reason: str | None = None
    error_type: str | None = None

    def __post_init__(self) -> None:
        _code(self.hypothesis_id, "cell hypothesis_id")
        spec = _FEATURE_BY_ID.get(self.feature_id)
        if spec is None:
            raise ObjectProfileError("cell feature_id is outside the frozen catalog")
        if not isinstance(self.state, ObjectFeatureCellState):
            raise TypeError("cell state must be ObjectFeatureCellState")
        reason = _optional_text(self.reason, "cell reason")
        error_type = _optional_text(self.error_type, "cell error_type")
        if self.state is ObjectFeatureCellState.SCORED:
            if not isinstance(self.interval, IntegerInterval):
                raise ObjectProfileError("scored cell requires an integer interval")
            if reason is not None or error_type is not None:
                raise ObjectProfileError("scored cell cannot carry failure fields")
        elif self.state is ObjectFeatureCellState.INDETERMINATE:
            if self.interval is not None or reason is None:
                raise ObjectProfileError(
                    "indeterminate cell requires a reason and no interval"
                )
            if error_type is not None:
                raise ObjectProfileError("indeterminate cell cannot carry error_type")
        else:
            if self.interval is not None or reason is None or error_type is None:
                raise ObjectProfileError(
                    "error cell requires reason and error_type, with no interval"
                )
        if self.interval is not None and spec.maximum is not None:
            if self.interval.upper > spec.maximum:
                raise ObjectProfileError("cell interval exceeds feature range")

    def to_data(self) -> dict[str, object]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "feature_id": self.feature_id,
            "state": self.state.value,
            "interval": None if self.interval is None else self.interval.to_data(),
            "reason": self.reason,
            "error_type": self.error_type,
        }

    @classmethod
    def from_data(cls, value: object) -> "ObjectFeatureCell":
        raw = _exact_fields(
            value,
            {"hypothesis_id", "feature_id", "state", "interval", "reason", "error_type"},
            "object feature cell",
        )
        try:
            state = ObjectFeatureCellState(raw["state"])
        except (TypeError, ValueError) as exc:
            raise ObjectProfileError("unknown object feature cell state") from exc
        interval = raw["interval"]
        return cls(
            hypothesis_id=raw["hypothesis_id"],
            feature_id=raw["feature_id"],
            state=state,
            interval=None if interval is None else IntegerInterval.from_data(interval),
            reason=raw["reason"],
            error_type=raw["error_type"],
        )


@dataclass(frozen=True, order=True, slots=True)
class ObjectProfileAtom:
    feature_id: str
    operator: ObjectProfileOperator
    target: int

    def __post_init__(self) -> None:
        spec = _FEATURE_BY_ID.get(self.feature_id)
        if spec is None:
            raise ObjectProfileError("profile atom feature is outside frozen catalog")
        if not isinstance(self.operator, ObjectProfileOperator):
            raise TypeError("profile operator must be ObjectProfileOperator")
        if self.operator not in spec.allowed_comparators:
            raise ObjectProfileError("operator is not allowed for this feature")
        _integer(self.target, "profile target", minimum=1)
        if spec.maximum is not None and self.target > spec.maximum:
            raise ObjectProfileError("profile target exceeds feature range")

    def to_data(self) -> dict[str, object]:
        return {
            "feature_id": self.feature_id,
            "operator": self.operator.value,
            "target": self.target,
        }

    @classmethod
    def from_data(cls, value: object) -> "ObjectProfileAtom":
        raw = _exact_fields(value, {"feature_id", "operator", "target"}, "profile atom")
        try:
            operator = ObjectProfileOperator(raw["operator"])
        except (TypeError, ValueError) as exc:
            raise ObjectProfileError("profile atom operator is not closed") from exc
        return cls(raw["feature_id"], operator, raw["target"])


def _profile_content(profile_id: str, atoms: tuple[ObjectProfileAtom, ...]) -> dict[str, object]:
    return {
        "schema": OBJECT_PROFILE_SCHEMA,
        "profile_id": profile_id,
        "formula": "all_atoms_on_one_hypothesis",
        "atoms": [item.to_data() for item in atoms],
        "catalog_digest": OBJECT_FEATURE_CATALOG_DIGEST,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectProfile:
    profile_id: str
    atoms: tuple[ObjectProfileAtom, ...]
    catalog_digest: str
    profile_digest: str

    def __post_init__(self) -> None:
        _code(self.profile_id, "profile_id")
        if not isinstance(self.atoms, tuple) or not self.atoms or any(
            not isinstance(item, ObjectProfileAtom) for item in self.atoms
        ):
            raise ObjectProfileError("profile atoms must be a nonempty tuple")
        indices = [_feature_index(item.feature_id) for item in self.atoms]
        if indices != sorted(indices) or len(indices) != len(set(indices)):
            raise ObjectProfileError("profile atoms must be unique and in catalog order")
        if self.catalog_digest != OBJECT_FEATURE_CATALOG_DIGEST:
            raise ObjectProfileError("profile catalog digest differs from frozen catalog")
        _digest(self.profile_digest, "profile_digest")
        if self.profile_digest != canonical_digest(_profile_content(self.profile_id, self.atoms)):
            raise ObjectProfileError("profile digest differs from canonical profile")

    @classmethod
    def create(
        cls, profile_id: str, atoms: Sequence[ObjectProfileAtom]
    ) -> "ObjectProfile":
        frozen = tuple(atoms)
        return cls(
            profile_id=profile_id,
            atoms=frozen,
            catalog_digest=OBJECT_FEATURE_CATALOG_DIGEST,
            profile_digest=canonical_digest(_profile_content(profile_id, frozen)),
        )

    def to_data(self) -> dict[str, object]:
        return {**_profile_content(self.profile_id, self.atoms), "profile_digest": self.profile_digest}

    @classmethod
    def from_data(cls, value: object) -> "ObjectProfile":
        expected = {
            "schema", "profile_id", "formula", "atoms", "catalog_digest",
            "profile_digest", *_authority_data(),
        }
        raw = _exact_fields(value, expected, "object profile")
        _validate_static_fields(raw, OBJECT_PROFILE_SCHEMA)
        if raw["formula"] != "all_atoms_on_one_hypothesis":
            raise ObjectProfileError("profile formula is not the closed conjunction")
        if not isinstance(raw["atoms"], list):
            raise ObjectProfileError("profile atoms must be a JSON list")
        return cls(
            profile_id=raw["profile_id"],
            atoms=tuple(ObjectProfileAtom.from_data(item) for item in raw["atoms"]),
            catalog_digest=raw["catalog_digest"],
            profile_digest=raw["profile_digest"],
        )


def _feature_index(feature_id: str) -> int:
    try:
        return OBJECT_FEATURE_IDS.index(feature_id)
    except ValueError as exc:
        raise ObjectProfileError("feature is outside frozen catalog") from exc


@dataclass(frozen=True, slots=True)
class ObjectHypothesisBinding:
    scenario_id: str
    hypothesis_id: str
    source_component_ids: tuple[str, ...]
    source_component_mask_digests: tuple[str, ...]
    union_mask_digest: str
    union_bbox: tuple[int, int, int, int]
    union_crop_digest: str
    hypothesis_catalog_digest: str
    feature_catalog_digest: str = OBJECT_FEATURE_CATALOG_DIGEST

    def __post_init__(self) -> None:
        _code(self.scenario_id, "binding scenario_id")
        _code(self.hypothesis_id, "hypothesis_id")
        if (
            not isinstance(self.source_component_ids, tuple)
            or not self.source_component_ids
            or tuple(sorted(self.source_component_ids)) != self.source_component_ids
            or len(set(self.source_component_ids)) != len(self.source_component_ids)
        ):
            raise ObjectProfileError("source component IDs must be nonempty, unique, and sorted")
        if any(_CODE.fullmatch(item) is None for item in self.source_component_ids):
            raise ObjectProfileError("source component ID must be a bounded code")
        if (
            not isinstance(self.source_component_mask_digests, tuple)
            or len(self.source_component_mask_digests) != len(self.source_component_ids)
        ):
            raise ObjectProfileError("component IDs and mask digests must align")
        for item in self.source_component_mask_digests:
            _digest(item, "source component mask digest")
        _digest(self.union_mask_digest, "union_mask_digest")
        _digest(self.union_crop_digest, "union_crop_digest")
        _digest(self.hypothesis_catalog_digest, "hypothesis_catalog_digest")
        if (
            not isinstance(self.union_bbox, tuple)
            or len(self.union_bbox) != 4
            or any(isinstance(item, bool) or not isinstance(item, int) for item in self.union_bbox)
        ):
            raise ObjectProfileError("union_bbox must be an integer (left, top, right, bottom) tuple")
        left, top, right, bottom = self.union_bbox
        if min(left, top) < 0 or right <= left or bottom <= top:
            raise ObjectProfileError("union_bbox must have positive area and nonnegative origin")
        if self.feature_catalog_digest != OBJECT_FEATURE_CATALOG_DIGEST:
            raise ObjectProfileError("hypothesis catalog digest differs from frozen catalog")

    def to_data(self) -> dict[str, object]:
        return {
            "scenario_id": self.scenario_id,
            "hypothesis_id": self.hypothesis_id,
            "source_component_ids": list(self.source_component_ids),
            "source_component_mask_digests": list(self.source_component_mask_digests),
            "union_mask_digest": self.union_mask_digest,
            "union_bbox": list(self.union_bbox),
            "union_crop_digest": self.union_crop_digest,
            "hypothesis_catalog_digest": self.hypothesis_catalog_digest,
            "feature_catalog_digest": self.feature_catalog_digest,
            "candidate_independent": True,
            "semantic_object_claimed": False,
        }

    @classmethod
    def from_data(cls, value: object) -> "ObjectHypothesisBinding":
        expected = {
            "scenario_id", "hypothesis_id", "source_component_ids", "source_component_mask_digests",
            "union_mask_digest", "union_bbox", "union_crop_digest",
            "hypothesis_catalog_digest", "feature_catalog_digest",
            "candidate_independent", "semantic_object_claimed",
        }
        raw = _exact_fields(value, expected, "object hypothesis binding")
        if raw["candidate_independent"] is not True or raw["semantic_object_claimed"] is not False:
            raise ObjectProfileError("hypothesis epistemic boundary differs")
        for name in ("source_component_ids", "source_component_mask_digests", "union_bbox"):
            if not isinstance(raw[name], list):
                raise ObjectProfileError(f"{name} must be a JSON list")
        return cls(
            scenario_id=raw["scenario_id"],
            hypothesis_id=raw["hypothesis_id"],
            source_component_ids=tuple(raw["source_component_ids"]),
            source_component_mask_digests=tuple(raw["source_component_mask_digests"]),
            union_mask_digest=raw["union_mask_digest"],
            union_bbox=tuple(raw["union_bbox"]),
            union_crop_digest=raw["union_crop_digest"],
            hypothesis_catalog_digest=raw["hypothesis_catalog_digest"],
            feature_catalog_digest=raw["feature_catalog_digest"],
        )


def _packet_content(
    scenario_id: str,
    panel_digest: str,
    visual_witness_packet_digest: str,
    hypothesis_catalog_digest: str,
    feature_protocol_digest: str,
    feature_model_id: str,
    feature_receipt_digest: str,
    feature_payload_digest: str,
    hypotheses: tuple[ObjectHypothesisBinding, ...],
    cells: tuple[ObjectFeatureCell, ...],
) -> dict[str, object]:
    return {
        "schema": OBJECT_LOCAL_OBSERVATION_PACKET_SCHEMA,
        "scenario_id": scenario_id,
        "panel_digest": panel_digest,
        "visual_witness_packet_digest": visual_witness_packet_digest,
        "hypothesis_catalog_digest": hypothesis_catalog_digest,
        "feature_protocol_digest": feature_protocol_digest,
        "feature_model_id": feature_model_id,
        "feature_receipt_digest": feature_receipt_digest,
        "feature_payload_digest": feature_payload_digest,
        "hypotheses": [item.to_data() for item in hypotheses],
        "cells": [item.to_data() for item in cells],
        "cell_order": "hypothesis_major_catalog_minor",
        "catalog_digest": OBJECT_FEATURE_CATALOG_DIGEST,
        "exhaustive": True,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectLocalObservationPacket:
    scenario_id: str
    panel_digest: str
    visual_witness_packet_digest: str
    hypothesis_catalog_digest: str
    feature_protocol_digest: str
    feature_model_id: str
    feature_receipt_digest: str
    feature_payload_digest: str
    hypotheses: tuple[ObjectHypothesisBinding, ...]
    cells: tuple[ObjectFeatureCell, ...]
    catalog_digest: str
    packet_digest: str

    def __post_init__(self) -> None:
        _code(self.scenario_id, "scenario_id")
        for label, value in (
            ("panel_digest", self.panel_digest),
            ("visual_witness_packet_digest", self.visual_witness_packet_digest),
            ("hypothesis_catalog_digest", self.hypothesis_catalog_digest),
            ("feature_protocol_digest", self.feature_protocol_digest),
            ("feature_receipt_digest", self.feature_receipt_digest),
            ("feature_payload_digest", self.feature_payload_digest),
        ):
            _digest(value, label)
        _code(self.feature_model_id, "feature_model_id")
        if not isinstance(self.hypotheses, tuple):
            raise ObjectProfileError("packet hypotheses must be a tuple")
        if any(not isinstance(item, ObjectHypothesisBinding) for item in self.hypotheses):
            raise ObjectProfileError("packet hypothesis has wrong type")
        ids = tuple(item.hypothesis_id for item in self.hypotheses)
        if ids != tuple(sorted(ids)) or len(ids) != len(set(ids)):
            raise ObjectProfileError("packet hypotheses must have unique sorted IDs")
        if not isinstance(self.cells, tuple) or any(
            not isinstance(item, ObjectFeatureCell) for item in self.cells
        ):
            raise ObjectProfileError("packet cells must be a tuple of ObjectFeatureCell")
        expected_order = tuple((hid, fid) for hid in ids for fid in OBJECT_FEATURE_IDS)
        actual_order = tuple((item.hypothesis_id, item.feature_id) for item in self.cells)
        if actual_order != expected_order:
            raise ObjectProfileError("packet is not exhaustive in exact hypothesis×feature order")
        if self.catalog_digest != OBJECT_FEATURE_CATALOG_DIGEST or any(
            item.feature_catalog_digest != self.catalog_digest
            or item.hypothesis_catalog_digest != self.hypothesis_catalog_digest
            or item.scenario_id != self.scenario_id
            for item in self.hypotheses
        ):
            raise ObjectProfileError("packet catalog binding differs from frozen catalog")
        _digest(self.packet_digest, "packet_digest")
        expected_digest = canonical_digest(
            _packet_content(
                self.scenario_id,
                self.panel_digest,
                self.visual_witness_packet_digest,
                self.hypothesis_catalog_digest,
                self.feature_protocol_digest,
                self.feature_model_id,
                self.feature_receipt_digest,
                self.feature_payload_digest,
                self.hypotheses,
                self.cells,
            )
        )
        if self.packet_digest != expected_digest:
            raise ObjectProfileError("packet digest differs from canonical packet")

    @classmethod
    def create(
        cls,
        scenario_id: str,
        hypotheses: Sequence[ObjectHypothesisBinding],
        cells: Sequence[ObjectFeatureCell],
        *,
        panel_digest: str,
        visual_witness_packet_digest: str,
        hypothesis_catalog_digest: str,
        feature_protocol_digest: str,
        feature_model_id: str,
        feature_receipt_digest: str,
        feature_payload_digest: str,
    ) -> "ObjectLocalObservationPacket":
        frozen_hypotheses, frozen_cells = tuple(hypotheses), tuple(cells)
        return cls(
            scenario_id=scenario_id,
            panel_digest=panel_digest,
            visual_witness_packet_digest=visual_witness_packet_digest,
            hypothesis_catalog_digest=hypothesis_catalog_digest,
            feature_protocol_digest=feature_protocol_digest,
            feature_model_id=feature_model_id,
            feature_receipt_digest=feature_receipt_digest,
            feature_payload_digest=feature_payload_digest,
            hypotheses=frozen_hypotheses,
            cells=frozen_cells,
            catalog_digest=OBJECT_FEATURE_CATALOG_DIGEST,
            packet_digest=canonical_digest(
                _packet_content(
                    scenario_id,
                    panel_digest,
                    visual_witness_packet_digest,
                    hypothesis_catalog_digest,
                    feature_protocol_digest,
                    feature_model_id,
                    feature_receipt_digest,
                    feature_payload_digest,
                    frozen_hypotheses,
                    frozen_cells,
                )
            ),
        )

    def to_data(self) -> dict[str, object]:
        return {
            **_packet_content(
                self.scenario_id,
                self.panel_digest,
                self.visual_witness_packet_digest,
                self.hypothesis_catalog_digest,
                self.feature_protocol_digest,
                self.feature_model_id,
                self.feature_receipt_digest,
                self.feature_payload_digest,
                self.hypotheses,
                self.cells,
            ),
            "packet_digest": self.packet_digest,
        }

    @classmethod
    def from_data(cls, value: object) -> "ObjectLocalObservationPacket":
        expected = {
            "schema", "scenario_id", "hypotheses", "cells", "cell_order",
            "catalog_digest", "exhaustive", "packet_digest", *_authority_data(),
            "panel_digest", "visual_witness_packet_digest", "hypothesis_catalog_digest",
            "feature_protocol_digest", "feature_model_id", "feature_receipt_digest",
            "feature_payload_digest",
        }
        raw = _exact_fields(value, expected, "object local observation packet")
        _validate_static_fields(raw, OBJECT_LOCAL_OBSERVATION_PACKET_SCHEMA)
        if raw["cell_order"] != "hypothesis_major_catalog_minor" or raw["exhaustive"] is not True:
            raise ObjectProfileError("packet coverage/order declaration differs")
        if not isinstance(raw["hypotheses"], list) or not isinstance(raw["cells"], list):
            raise ObjectProfileError("packet hypotheses and cells must be JSON lists")
        return cls(
            scenario_id=raw["scenario_id"],
            panel_digest=raw["panel_digest"],
            visual_witness_packet_digest=raw["visual_witness_packet_digest"],
            hypothesis_catalog_digest=raw["hypothesis_catalog_digest"],
            feature_protocol_digest=raw["feature_protocol_digest"],
            feature_model_id=raw["feature_model_id"],
            feature_receipt_digest=raw["feature_receipt_digest"],
            feature_payload_digest=raw["feature_payload_digest"],
            hypotheses=tuple(ObjectHypothesisBinding.from_data(item) for item in raw["hypotheses"]),
            cells=tuple(ObjectFeatureCell.from_data(item) for item in raw["cells"]),
            catalog_digest=raw["catalog_digest"],
            packet_digest=raw["packet_digest"],
        )


def _validate_static_fields(raw: Mapping[str, Any], schema: str) -> None:
    if raw["schema"] != schema:
        raise ObjectProfileError("schema identifier differs")
    for key, expected in _authority_data().items():
        if raw[key] != expected or type(raw[key]) is not type(expected):
            raise ObjectProfileError(f"authority field {key} differs")


@dataclass(frozen=True, slots=True)
class ObjectAtomEvaluation:
    feature_id: str
    disposition: Disposition

    def to_data(self) -> dict[str, str]:
        return {"feature_id": self.feature_id, "disposition": self.disposition.value}


@dataclass(frozen=True, slots=True)
class ObjectHypothesisEvaluation:
    hypothesis_id: str
    atoms: tuple[ObjectAtomEvaluation, ...]
    disposition: Disposition

    def to_data(self) -> dict[str, object]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "atoms": [item.to_data() for item in self.atoms],
            "disposition": self.disposition.value,
        }


@dataclass(frozen=True, slots=True)
class ObjectScenarioEvaluation:
    scenario_id: str
    packet_digest: str
    hypotheses: tuple[ObjectHypothesisEvaluation, ...]
    disposition: Disposition

    def to_data(self) -> dict[str, object]:
        return {
            "scenario_id": self.scenario_id,
            "packet_digest": self.packet_digest,
            "hypotheses": [item.to_data() for item in self.hypotheses],
            "disposition": self.disposition.value,
        }


def _evaluation_content(
    profile: ObjectProfile, scenarios: tuple[ObjectScenarioEvaluation, ...], disposition: Disposition
) -> dict[str, object]:
    return {
        "schema": OBJECT_PROFILE_EVALUATION_SCHEMA,
        "profile_digest": profile.profile_digest,
        "catalog_digest": OBJECT_FEATURE_CATALOG_DIGEST,
        "scenario_packet_digests": [item.packet_digest for item in scenarios],
        "scenarios": [item.to_data() for item in scenarios],
        "disposition": disposition.value,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectProfileEvaluation:
    profile_digest: str
    catalog_digest: str
    scenario_packet_digests: tuple[str, ...]
    scenarios: tuple[ObjectScenarioEvaluation, ...]
    disposition: Disposition
    evaluation_digest: str

    def __post_init__(self) -> None:
        self._validate()

    def to_data(self) -> dict[str, object]:
        content = {
            "schema": OBJECT_PROFILE_EVALUATION_SCHEMA,
            "profile_digest": self.profile_digest,
            "catalog_digest": self.catalog_digest,
            "scenario_packet_digests": list(self.scenario_packet_digests),
            "scenarios": [item.to_data() for item in self.scenarios],
            "disposition": self.disposition.value,
            **_authority_data(),
        }
        return {**content, "evaluation_digest": self.evaluation_digest}

    @classmethod
    def from_data(cls, value: object) -> "ObjectProfileEvaluation":
        expected = {
            "schema", "profile_digest", "catalog_digest", "scenario_packet_digests",
            "scenarios", "disposition", "evaluation_digest", *_authority_data(),
        }
        raw = _exact_fields(value, expected, "object profile evaluation")
        _validate_static_fields(raw, OBJECT_PROFILE_EVALUATION_SCHEMA)
        if not isinstance(raw["scenario_packet_digests"], list) or not isinstance(raw["scenarios"], list):
            raise ObjectProfileError("evaluation scenario fields must be JSON lists")
        scenarios = tuple(_scenario_evaluation_from_data(item) for item in raw["scenarios"])
        try:
            disposition = Disposition(raw["disposition"])
        except (TypeError, ValueError) as exc:
            raise ObjectProfileError("unknown evaluation disposition") from exc
        result = cls(
            profile_digest=raw["profile_digest"],
            catalog_digest=raw["catalog_digest"],
            scenario_packet_digests=tuple(raw["scenario_packet_digests"]),
            scenarios=scenarios,
            disposition=disposition,
            evaluation_digest=raw["evaluation_digest"],
        )
        result._validate()
        return result

    def _validate(self) -> None:
        _digest(self.profile_digest, "evaluation profile_digest")
        if not isinstance(self.disposition, Disposition):
            raise TypeError("evaluation disposition must be a Disposition")
        if not isinstance(self.scenarios, tuple) or any(
            not isinstance(item, ObjectScenarioEvaluation) for item in self.scenarios
        ):
            raise ObjectProfileError("evaluation scenarios must be a typed tuple")
        if not isinstance(self.scenario_packet_digests, tuple):
            raise ObjectProfileError("evaluation packet digests must be a tuple")
        if self.catalog_digest != OBJECT_FEATURE_CATALOG_DIGEST:
            raise ObjectProfileError("evaluation catalog digest differs")
        if not self.scenarios or self.scenario_packet_digests != tuple(item.packet_digest for item in self.scenarios):
            raise ObjectProfileError("evaluation scenario packet binding differs")
        if tuple(item.scenario_id for item in self.scenarios) != tuple(sorted(item.scenario_id for item in self.scenarios)):
            raise ObjectProfileError("evaluation scenarios are not in exact order")
        for item in self.scenario_packet_digests:
            _digest(item, "evaluation packet digest")
        _digest(self.evaluation_digest, "evaluation_digest")
        content = self.to_data().copy()
        content.pop("evaluation_digest")
        if self.evaluation_digest != canonical_digest(content):
            raise ObjectProfileError("evaluation digest differs from canonical evaluation")
        expected = _combine_scenarios(tuple(item.disposition for item in self.scenarios))
        if self.disposition is not expected:
            raise ObjectProfileError("evaluation disposition differs from scenario replay")


def _atom_evaluation_from_data(value: object) -> ObjectAtomEvaluation:
    raw = _exact_fields(value, {"feature_id", "disposition"}, "atom evaluation")
    if raw["feature_id"] not in _FEATURE_BY_ID:
        raise ObjectProfileError("atom evaluation feature differs from catalog")
    try:
        return ObjectAtomEvaluation(raw["feature_id"], Disposition(raw["disposition"]))
    except (TypeError, ValueError) as exc:
        raise ObjectProfileError("unknown atom evaluation disposition") from exc


def _hypothesis_evaluation_from_data(value: object) -> ObjectHypothesisEvaluation:
    raw = _exact_fields(value, {"hypothesis_id", "atoms", "disposition"}, "hypothesis evaluation")
    if not isinstance(raw["atoms"], list):
        raise ObjectProfileError("hypothesis evaluation atoms must be a JSON list")
    atoms = tuple(_atom_evaluation_from_data(item) for item in raw["atoms"])
    try:
        disposition = Disposition(raw["disposition"])
    except (TypeError, ValueError) as exc:
        raise ObjectProfileError("unknown hypothesis disposition") from exc
    expected = _combine_atoms(tuple(item.disposition for item in atoms))
    if disposition is not expected:
        raise ObjectProfileError("hypothesis disposition differs from atom replay")
    return ObjectHypothesisEvaluation(raw["hypothesis_id"], atoms, disposition)


def _scenario_evaluation_from_data(value: object) -> ObjectScenarioEvaluation:
    raw = _exact_fields(value, {"scenario_id", "packet_digest", "hypotheses", "disposition"}, "scenario evaluation")
    if not isinstance(raw["hypotheses"], list):
        raise ObjectProfileError("scenario hypotheses must be a JSON list")
    hypotheses = tuple(_hypothesis_evaluation_from_data(item) for item in raw["hypotheses"])
    try:
        disposition = Disposition(raw["disposition"])
    except (TypeError, ValueError) as exc:
        raise ObjectProfileError("unknown scenario disposition") from exc
    expected = _combine_hypotheses(tuple(item.disposition for item in hypotheses))
    if disposition is not expected:
        raise ObjectProfileError("scenario disposition differs from hypothesis replay")
    return ObjectScenarioEvaluation(raw["scenario_id"], _digest(raw["packet_digest"], "packet_digest"), hypotheses, disposition)


def _combine_atoms(values: tuple[Disposition, ...]) -> Disposition:
    if any(item is Disposition.CERTIFIED_ABSENT for item in values):
        return Disposition.CERTIFIED_ABSENT
    if any(item is Disposition.ERROR for item in values):
        return Disposition.ERROR
    if any(item is Disposition.INDETERMINATE for item in values):
        return Disposition.INDETERMINATE
    return Disposition.PRESENT


def _combine_hypotheses(values: tuple[Disposition, ...]) -> Disposition:
    if any(item is Disposition.PRESENT for item in values):
        return Disposition.PRESENT
    if all(item is Disposition.CERTIFIED_ABSENT for item in values):
        return Disposition.CERTIFIED_ABSENT
    if any(item is Disposition.ERROR for item in values):
        return Disposition.ERROR
    return Disposition.INDETERMINATE


def _combine_scenarios(values: tuple[Disposition, ...]) -> Disposition:
    if any(item is Disposition.ERROR for item in values):
        return Disposition.ERROR
    if all(item is Disposition.PRESENT for item in values):
        return Disposition.PRESENT
    if all(item is Disposition.CERTIFIED_ABSENT for item in values):
        return Disposition.CERTIFIED_ABSENT
    return Disposition.INDETERMINATE


def _evaluate_atom(atom: ObjectProfileAtom, cell: ObjectFeatureCell) -> Disposition:
    if cell.state is ObjectFeatureCellState.ERROR:
        return Disposition.ERROR
    if cell.state is ObjectFeatureCellState.INDETERMINATE:
        return Disposition.INDETERMINATE
    assert cell.interval is not None
    lower, upper = cell.interval.lower, cell.interval.upper
    if atom.operator is ObjectProfileOperator.EQUALS:
        if lower == upper == atom.target:
            return Disposition.PRESENT
        if atom.target < lower or atom.target > upper:
            return Disposition.CERTIFIED_ABSENT
        return Disposition.INDETERMINATE
    if lower >= atom.target:
        return Disposition.PRESENT
    if upper < atom.target:
        return Disposition.CERTIFIED_ABSENT
    return Disposition.INDETERMINATE


def evaluate_object_profile(
    profile: ObjectProfile,
    packets: Sequence[ObjectLocalObservationPacket],
) -> ObjectProfileEvaluation:
    """Evaluate one closed conjunction existentially per scenario, then unanimously."""

    if not isinstance(profile, ObjectProfile):
        raise TypeError("profile must be ObjectProfile")
    # Replay all self-authenticating invariants, including digests.
    profile = ObjectProfile.from_data(profile.to_data())
    frozen_packets = tuple(packets)
    if not frozen_packets or any(not isinstance(item, ObjectLocalObservationPacket) for item in frozen_packets):
        raise ObjectProfileError("packets must be a nonempty sequence of observation packets")
    frozen_packets = tuple(ObjectLocalObservationPacket.from_data(item.to_data()) for item in frozen_packets)
    scenario_ids = tuple(item.scenario_id for item in frozen_packets)
    if scenario_ids != tuple(sorted(scenario_ids)) or len(set(scenario_ids)) != len(scenario_ids):
        raise ObjectProfileError("packets must have unique scenario IDs in exact sorted order")

    scenarios: list[ObjectScenarioEvaluation] = []
    for packet in frozen_packets:
        by_key = {(item.hypothesis_id, item.feature_id): item for item in packet.cells}
        hypotheses: list[ObjectHypothesisEvaluation] = []
        for binding in packet.hypotheses:
            atoms = tuple(
                ObjectAtomEvaluation(
                    atom.feature_id,
                    _evaluate_atom(atom, by_key[(binding.hypothesis_id, atom.feature_id)]),
                )
                for atom in profile.atoms
            )
            hypotheses.append(
                ObjectHypothesisEvaluation(
                    binding.hypothesis_id,
                    atoms,
                    _combine_atoms(tuple(item.disposition for item in atoms)),
                )
            )
        frozen_hypotheses = tuple(hypotheses)
        scenarios.append(
            ObjectScenarioEvaluation(
                packet.scenario_id,
                packet.packet_digest,
                frozen_hypotheses,
                _combine_hypotheses(tuple(item.disposition for item in frozen_hypotheses)),
            )
        )
    frozen_scenarios = tuple(scenarios)
    disposition = _combine_scenarios(tuple(item.disposition for item in frozen_scenarios))
    content = _evaluation_content(profile, frozen_scenarios, disposition)
    result = ObjectProfileEvaluation(
        profile_digest=profile.profile_digest,
        catalog_digest=OBJECT_FEATURE_CATALOG_DIGEST,
        scenario_packet_digests=tuple(item.packet_digest for item in frozen_scenarios),
        scenarios=frozen_scenarios,
        disposition=disposition,
        evaluation_digest=canonical_digest(content),
    )
    result._validate()
    return result


def verify_object_profile_evaluation(
    evaluation: ObjectProfileEvaluation,
    *,
    profile: ObjectProfile,
    packets: Sequence[ObjectLocalObservationPacket],
) -> ObjectProfileEvaluation:
    """Cold-replay an evaluation against its exact profile and packet objects."""

    if not isinstance(evaluation, ObjectProfileEvaluation):
        raise TypeError("evaluation must be ObjectProfileEvaluation")
    decoded = ObjectProfileEvaluation.from_data(evaluation.to_data())
    expected = evaluate_object_profile(profile, packets)
    if decoded != expected:
        raise ObjectProfileError("evaluation differs from exact profile/packet replay")
    return decoded


__all__ = (
    "IntegerInterval",
    "OBJECT_FEATURE_CATALOG",
    "OBJECT_FEATURE_CATALOG_DIGEST",
    "OBJECT_FEATURE_IDS",
    "ObjectAtomEvaluation",
    "ObjectFeatureCell",
    "ObjectFeatureCellState",
    "ObjectFeatureSpec",
    "ObjectHypothesisBinding",
    "ObjectHypothesisEvaluation",
    "ObjectLocalObservationPacket",
    "ObjectProfile",
    "ObjectProfileAtom",
    "ObjectProfileError",
    "ObjectProfileEvaluation",
    "ObjectProfileOperator",
    "ObjectScenarioEvaluation",
    "evaluate_object_profile",
    "object_feature_catalog_data",
    "object_feature_catalog_digest",
    "verify_object_profile_evaluation",
)

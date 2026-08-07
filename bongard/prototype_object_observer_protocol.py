"""Closed vision protocol for reference profiles and object-local features.

The reference turn may choose atoms from one frozen catalog.  The later scene
turn is deliberately blind to those choices: it receives only opaque atlas
slots and the complete catalog, and must report every slot-by-feature cell.
Python validates and evaluates those cells; model prose is audit evidence only.
"""

from __future__ import annotations

from bongard.runtime_source_snapshot import capture_loaded_source, verify_loaded_source


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)

from dataclasses import dataclass
from enum import Enum
import hashlib
import re
from typing import Any, Mapping, Sequence

from bongard.canonical import canonical_digest
from bongard.prototype_object_hypotheses import (
    ObjectHypothesisPacket,
    object_hypothesis_extractor_artifact_digest,
)
from bongard.prototype_object_profiles import (
    OBJECT_FEATURE_CATALOG,
    OBJECT_FEATURE_CATALOG_DIGEST,
    OBJECT_FEATURE_IDS,
    IntegerInterval,
    ObjectFeatureCell,
    ObjectFeatureCellState,
    ObjectHypothesisBinding,
    ObjectLocalObservationPacket,
    ObjectProfile,
    ObjectProfileAtom,
    ObjectProfileOperator,
)
from bongard.prototype_visual_runtime import visual_runtime_dependency_digest
from bongard.python_predicate_authority import PYTHON_PREDICATE_AUTHORITY_ID


DESCRIPTION_PROTOCOL_ID = "bongard.prototype-object-observer/reference-profile-v1"
FEATURE_PROTOCOL_ID = "bongard.prototype-object-observer/profile-blind-atlas-v1"
DESCRIPTION_SCHEMA_ID = "gkm.bongard-object-profile-description-payload.v1"
FEATURE_SCHEMA_ID = "gkm.bongard-object-feature-observation-payload.v1"
PROTOTYPE_GROUP_IDS = ("group_0", "group_1")

_PROSE = re.compile(r"[A-Za-z0-9][A-Za-z0-9 ,.\'-]{0,767}\Z")
_REASON = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")
_FORBIDDEN_PROSE = re.compile(
    r"\b(?:task|label|query|prompt|instruction|system|assistant|user|tool|"
    r"code|python|lean|theorem|predicate|positive|negative|class)s?\b",
    re.IGNORECASE,
)


class PrototypeObjectProtocolError(ValueError):
    """A payload or protocol identity violates the frozen finite grammar."""


def _source_digest() -> str:
    return verify_loaded_source(__name__, expected_source_sha256=_LOADED_SOURCE_SHA256)


def prototype_object_protocol_source_digest() -> str:
    """Public import-time source identity for precommit and cold replay."""

    return _source_digest()


def _authority_data() -> dict[str, object]:
    return {
        "predicate_authority_id": PYTHON_PREDICATE_AUTHORITY_ID,
        "python_is_canonical_authority": True,
        "lean_present": False,
        "lean_required": False,
        "lean_removable": True,
        "lean_affects_identity_or_decision": False,
    }


def _exact(value: object, fields: set[str], label: str) -> Mapping[str, Any]:
    if (
        not isinstance(value, Mapping)
        or any(not isinstance(key, str) for key in value)
        or set(value) != fields
    ):
        raise PrototypeObjectProtocolError(f"{label} fields differ from schema")
    return value


def _rows(value: object, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise PrototypeObjectProtocolError(f"{label} must be a JSON list")
    return value


def _audit_prose(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or value != value.strip()
        or not value
        or len(value.encode("utf-8")) > 768
        or _PROSE.fullmatch(value) is None
        or _FORBIDDEN_PROSE.search(value) is not None
    ):
        raise PrototypeObjectProtocolError(f"{label} violates neutral prose policy")
    return value


class ObjectAuditTextState(str, Enum):
    DEFINED = "defined"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class ObjectAuditText:
    state: ObjectAuditTextState
    prose: str | None

    @classmethod
    def defined(cls, prose: str) -> "ObjectAuditText":
        return cls(ObjectAuditTextState.DEFINED, _audit_prose(prose, "audit prose"))

    @classmethod
    def rejected(cls) -> "ObjectAuditText":
        return cls(ObjectAuditTextState.REJECTED, None)


@dataclass(frozen=True, slots=True)
class ParsedPrototypeObjectDescription:
    audit_rubrics: tuple[ObjectAuditText, ObjectAuditText]
    profiles: tuple[ObjectProfile, ObjectProfile]


@dataclass(frozen=True, slots=True)
class ParsedPrototypeObjectFeatures:
    audit_description: ObjectAuditText
    packets: tuple[ObjectLocalObservationPacket, ...]


def _feature_catalog_lines() -> str:
    values: list[str] = []
    for item in OBJECT_FEATURE_CATALOG:
        operators = ",".join(value.value for value in item.allowed_comparators)
        maximum = "unbounded" if item.maximum is None else str(item.maximum)
        values.append(
            f"- {item.feature_id}; unit={item.unit}; range=0..{maximum}; "
            f"profile operators={operators}; meaning={item.operational_description}"
        )
    return "\n".join(values)


def prototype_object_description_prompt() -> str:
    return (
        "Inspect six reference images in two neutral groups of three. For each "
        "group, write one concise sentence describing the recurring visible "
        "appearance, then express that description using one or more atoms "
        "from the complete frozen measurement catalog below. Ignore pose, "
        "scale, location, and incidental stroke variation. Atoms are a "
        "same-object conjunction. Use only the declared feature identifiers, "
        "operators, and strictly positive integer targets. Return group_0 then "
        "group_1. Do not emit executable text, negation, disjunction, weights, "
        "or hidden-role guesses. Atom order is ignored and canonicalized by "
        "the Python verifier.\n\nFrozen measurement catalog:\n"
        + _feature_catalog_lines()
    )


def prototype_object_description_output_schema() -> dict[str, object]:
    atom = {
        "type": "object",
        "properties": {
            "feature_id": {"type": "string", "enum": list(OBJECT_FEATURE_IDS)},
            "operator": {
                "type": "string",
                "enum": [value.value for value in ObjectProfileOperator],
            },
            "target": {"type": "integer"},
        },
        "required": ["feature_id", "operator", "target"],
        "additionalProperties": False,
    }
    row = {
        "type": "object",
        "properties": {
            "group_id": {"type": "string", "enum": list(PROTOTYPE_GROUP_IDS)},
            "rubric": {"type": "string"},
            "atoms": {"type": "array", "items": atom},
        },
        "required": ["group_id", "rubric", "atoms"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {"profiles": {"type": "array", "items": row}},
        "required": ["profiles"],
        "additionalProperties": False,
    }


def prototype_object_description_protocol_digest() -> str:
    prompt = prototype_object_description_prompt()
    schema = prototype_object_description_output_schema()
    return canonical_digest(
        {
            "schema": DESCRIPTION_SCHEMA_ID,
            "protocol_id": DESCRIPTION_PROTOCOL_ID,
            "source_digest": _source_digest(),
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "output_schema_digest": canonical_digest(schema),
            "feature_catalog_digest": OBJECT_FEATURE_CATALOG_DIGEST,
            "group_order": list(PROTOTYPE_GROUP_IDS),
            "grammar": "positive-atoms-all-on-one-hypothesis",
            **_authority_data(),
        }
    )


def parse_prototype_object_description_payload(
    payload: object,
) -> ParsedPrototypeObjectDescription:
    raw = _exact(payload, {"profiles"}, "description payload")
    values = _rows(raw["profiles"], "description profiles")
    if len(values) != 2:
        raise PrototypeObjectProtocolError("description must exhaust two groups")
    audits: list[ObjectAuditText] = []
    profiles: list[ObjectProfile] = []
    for index, (value, group_id) in enumerate(
        zip(values, PROTOTYPE_GROUP_IDS, strict=True)
    ):
        row = _exact(value, {"group_id", "rubric", "atoms"}, f"profile {index}")
        if row["group_id"] != group_id:
            raise PrototypeObjectProtocolError("description group order differs")
        audits.append(ObjectAuditText.defined(_audit_prose(row["rubric"], "rubric")))
        atom_rows = _rows(row["atoms"], "profile atoms")
        if not atom_rows:
            raise PrototypeObjectProtocolError("profile atoms cannot be empty")
        atoms = tuple(ObjectProfileAtom.from_data(value) for value in atom_rows)
        atoms = tuple(
            sorted(
                atoms,
                key=lambda atom: OBJECT_FEATURE_IDS.index(atom.feature_id),
            )
        )
        profiles.append(ObjectProfile.create(group_id, atoms))
    return ParsedPrototypeObjectDescription(
        tuple(audits),  # type: ignore[arg-type]
        tuple(profiles),  # type: ignore[arg-type]
    )


def _slot_rows(packet: ObjectHypothesisPacket) -> tuple[tuple[str, int, int, str], ...]:
    return tuple(
        (sheet.name, slot.row_index, slot.column_index, slot.slot_id)
        for sheet in packet.atlas_sheets
        for slot in sheet.slots
    )


def prototype_object_feature_prompt(packet: ObjectHypothesisPacket) -> str:
    if not isinstance(packet, ObjectHypothesisPacket):
        raise TypeError("packet must be ObjectHypothesisPacket")
    slots = _slot_rows(packet)
    rendered_slots = (
        "- no occupied slots"
        if not slots
        else "\n".join(
            f"- {name}; row={row}; column={column}; slot_id={slot_id}"
            for name, row, column, slot_id in slots
        )
    )
    return (
        "Inspect the opaque contact sheets. Each occupied grid cell is an "
        "independent visual hypothesis. Measure every occupied slot against "
        "every feature in the complete frozen catalog below, in the exact "
        "slot-major then catalog-minor order. The grid uses zero-based rows "
        "and columns. For a scored cell return a closed integer uncertainty "
        "interval. Use indeterminate only when no honest interval can be "
        "given. Extra shapes elsewhere are irrelevant because each slot is "
        "isolated. Return one neutral scene sentence for audit only. No chosen "
        "group description or rule is available in this turn.\n\n"
        "Occupied slots:\n"
        + rendered_slots
        + "\n\nComplete frozen measurement catalog:\n"
        + _feature_catalog_lines()
    )


def prototype_object_feature_output_schema() -> dict[str, object]:
    nullable_integer = {"anyOf": [{"type": "integer"}, {"type": "null"}]}
    nullable_string = {"anyOf": [{"type": "string"}, {"type": "null"}]}
    cell = {
        "type": "object",
        "properties": {
            "slot_id": {"type": "string"},
            "feature_id": {"type": "string", "enum": list(OBJECT_FEATURE_IDS)},
            "state": {"type": "string", "enum": ["scored", "indeterminate"]},
            "lower": nullable_integer,
            "upper": nullable_integer,
            "reason_code": nullable_string,
        },
        "required": [
            "slot_id",
            "feature_id",
            "state",
            "lower",
            "upper",
            "reason_code",
        ],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "description": {"type": "string"},
            "cells": {"type": "array", "items": cell},
        },
        "required": ["description", "cells"],
        "additionalProperties": False,
    }


def prototype_object_feature_protocol_digest(packet: ObjectHypothesisPacket) -> str:
    prompt = prototype_object_feature_prompt(packet)
    schema = prototype_object_feature_output_schema()
    return canonical_digest(
        {
            "schema": FEATURE_SCHEMA_ID,
            "protocol_id": FEATURE_PROTOCOL_ID,
            "protocol_family_digest": prototype_object_feature_protocol_family_digest(),
            "hypothesis_packet_digest": packet.digest(),
            "hypothesis_extractor_digest": object_hypothesis_extractor_artifact_digest(),
            "visual_runtime_dependency_digest": visual_runtime_dependency_digest(),
            "feature_catalog_digest": OBJECT_FEATURE_CATALOG_DIGEST,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "output_schema_digest": canonical_digest(schema),
            "profile_blind": True,
            "reference_blind": True,
            "coverage": "every-occupied-slot-by-every-feature-in-exact-order",
            **_authority_data(),
        }
    )


def prototype_object_feature_protocol_family_digest() -> str:
    """Packet-independent identity of the profile-blind feature protocol."""

    return canonical_digest(
        {
            "schema": "gkm.bongard-object-feature-protocol-family.v1",
            "protocol_id": FEATURE_PROTOCOL_ID,
            "source_digest": _source_digest(),
            "output_schema_digest": canonical_digest(
                prototype_object_feature_output_schema()
            ),
            "feature_catalog_digest": OBJECT_FEATURE_CATALOG_DIGEST,
            "hypothesis_extractor_digest": object_hypothesis_extractor_artifact_digest(),
            "visual_runtime_dependency_digest": visual_runtime_dependency_digest(),
            "profile_blind": True,
            "reference_blind": True,
            "coverage": "every-occupied-slot-by-every-feature-in-exact-order",
            **_authority_data(),
        }
    )


def _reason(value: object) -> str:
    if not isinstance(value, str) or _REASON.fullmatch(value) is None:
        raise PrototypeObjectProtocolError("indeterminate reason is not bounded")
    return value


def parse_prototype_object_feature_payload(
    packet: ObjectHypothesisPacket,
    payload: object,
    *,
    feature_model_id: str,
    feature_receipt_digest: str,
) -> ParsedPrototypeObjectFeatures:
    if not isinstance(packet, ObjectHypothesisPacket):
        raise TypeError("packet must be ObjectHypothesisPacket")
    raw = _exact(payload, {"description", "cells"}, "feature payload")
    try:
        audit = ObjectAuditText.defined(_audit_prose(raw["description"], "scene description"))
    except (PrototypeObjectProtocolError, TypeError, ValueError):
        audit = ObjectAuditText.rejected()
    values = _rows(raw["cells"], "feature cells")
    slots = tuple(slot for sheet in packet.atlas_sheets for slot in sheet.slots)
    expected = tuple(
        (slot.slot_id, feature_id)
        for slot in slots
        for feature_id in OBJECT_FEATURE_IDS
    )
    if len(values) != len(expected):
        raise PrototypeObjectProtocolError("feature payload coverage differs")
    parsed_by_slot: dict[str, list[ObjectFeatureCell]] = {
        slot.slot_id: [] for slot in slots
    }
    for index, (value, (slot_id, feature_id)) in enumerate(
        zip(values, expected, strict=True)
    ):
        row = _exact(
            value,
            {"slot_id", "feature_id", "state", "lower", "upper", "reason_code"},
            f"feature cell {index}",
        )
        if (row["slot_id"], row["feature_id"]) != (slot_id, feature_id):
            raise PrototypeObjectProtocolError("feature cell order differs")
        hypothesis_id = next(
            slot.hypothesis_id for slot in slots if slot.slot_id == slot_id
        )
        if row["state"] == "scored":
            if row["reason_code"] is not None:
                raise PrototypeObjectProtocolError("scored cell carries reason")
            try:
                interval = IntegerInterval(row["lower"], row["upper"])
                cell = ObjectFeatureCell(
                    hypothesis_id,
                    feature_id,
                    ObjectFeatureCellState.SCORED,
                    interval,
                )
            except (TypeError, ValueError) as exc:
                raise PrototypeObjectProtocolError("scored interval is invalid") from exc
        elif row["state"] == "indeterminate":
            if row["lower"] is not None or row["upper"] is not None:
                raise PrototypeObjectProtocolError("indeterminate cell carries interval")
            cell = ObjectFeatureCell(
                hypothesis_id,
                feature_id,
                ObjectFeatureCellState.INDETERMINATE,
                None,
                _reason(row["reason_code"]),
            )
        else:
            raise PrototypeObjectProtocolError("unknown feature cell state")
        parsed_by_slot[slot_id].append(cell)

    hypothesis_digest = packet.digest()
    payload_digest = canonical_digest(dict(raw))
    protocol_digest = prototype_object_feature_protocol_digest(packet)
    slot_by_hypothesis = {
        (slot.scenario_id, slot.hypothesis_id): slot for slot in slots
    }
    packets: list[ObjectLocalObservationPacket] = []
    for scenario in packet.scenarios:
        bindings: list[ObjectHypothesisBinding] = []
        cells: list[ObjectFeatureCell] = []
        for hypothesis in scenario.hypotheses:
            slot = slot_by_hypothesis[(scenario.scenario_id, hypothesis.hypothesis_id)]
            bindings.append(
                ObjectHypothesisBinding(
                    scenario_id=scenario.scenario_id,
                    hypothesis_id=hypothesis.hypothesis_id,
                    source_component_ids=hypothesis.source_component_ids,
                    source_component_mask_digests=hypothesis.source_component_mask_digests,
                    union_mask_digest=hypothesis.union_mask_digest,
                    union_bbox=hypothesis.bbox_pixels,
                    union_crop_digest=hypothesis.masked_crop_pixel_digest,
                    hypothesis_catalog_digest=hypothesis_digest,
                )
            )
            cells.extend(parsed_by_slot[slot.slot_id])
        packets.append(
            ObjectLocalObservationPacket.create(
                scenario.scenario_id,
                bindings,
                cells,
                panel_digest=packet.panel_digest,
                visual_witness_packet_digest=packet.visual_witness_packet_digest,
                hypothesis_catalog_digest=hypothesis_digest,
                feature_protocol_digest=protocol_digest,
                feature_model_id=feature_model_id,
                feature_receipt_digest=feature_receipt_digest,
                feature_payload_digest=payload_digest,
            )
        )
    return ParsedPrototypeObjectFeatures(audit, tuple(packets))


__all__ = (
    "DESCRIPTION_PROTOCOL_ID",
    "FEATURE_PROTOCOL_ID",
    "ObjectAuditText",
    "ObjectAuditTextState",
    "ParsedPrototypeObjectDescription",
    "ParsedPrototypeObjectFeatures",
    "PrototypeObjectProtocolError",
    "parse_prototype_object_description_payload",
    "parse_prototype_object_feature_payload",
    "prototype_object_description_output_schema",
    "prototype_object_description_prompt",
    "prototype_object_description_protocol_digest",
    "prototype_object_feature_output_schema",
    "prototype_object_feature_prompt",
    "prototype_object_feature_protocol_family_digest",
    "prototype_object_feature_protocol_digest",
    "prototype_object_protocol_source_digest",
)

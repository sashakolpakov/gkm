"""Closed vision protocol for feature nominations and object-local evidence.

The reference turn may emit prose and nominate feature IDs only.  Python owns
the fixed operator/target operationalization.  The later scene turn is blind
to those nominations: it receives one bounded opaque-atlas shard and one slice
of the full frozen catalog, and exhaustive shards cover every slot/feature
pair.  Python validates and evaluates cells; model prose is audit evidence.
"""

from __future__ import annotations

from bongard.runtime_source_snapshot import capture_loaded_source, verify_loaded_source


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)

from dataclasses import dataclass
from enum import Enum
import hashlib
import re
from typing import Any, Mapping, Sequence

from bongard.canonical import canonical_digest, canonical_json
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


DESCRIPTION_PROTOCOL_ID = "bongard.prototype-object-observer/reference-feature-family-v2"
FEATURE_PROTOCOL_ID = "bongard.prototype-object-observer/profile-blind-atlas-shards-v2"
DESCRIPTION_SCHEMA_ID = "gkm.bongard-object-feature-family-description-payload.v2"
FEATURE_SCHEMA_ID = "gkm.bongard-object-feature-observation-matrix.v2"
PROTOTYPE_GROUP_IDS = ("group_0", "group_1")
MAX_FEATURE_SHARD_SLOTS = 16
MAX_FEATURE_SHARD_FEATURES = 15
MAX_FEATURE_SHARD_CELLS = MAX_FEATURE_SHARD_SLOTS * MAX_FEATURE_SHARD_FEATURES
MAX_FEATURE_SHARD_PAYLOAD_BYTES = 8_192
MAX_OBSERVED_COUNT = 16_777_216
MODEL_INDETERMINATE_REASON = "model_indeterminate"
DESCRIPTION_SUPPORT_TARGET_PPM = 500_000
DESCRIPTION_COUNT_TARGET = 1

_PROSE = re.compile(r"[A-Za-z0-9][A-Za-z0-9 ,.\'-]{0,767}\Z")
_REASON = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
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
    feature_families: tuple[tuple[str, ...], tuple[str, ...]]
    profiles: tuple[ObjectProfile, ObjectProfile]


@dataclass(frozen=True, slots=True)
class ParsedPrototypeObjectFeatures:
    audit_description: ObjectAuditText
    packets: tuple[ObjectLocalObservationPacket, ...]


class ObjectFeatureShardStatus(str, Enum):
    SUCCESS = "success"
    PARSER_ERROR = "parser_error"
    TRANSPORT_ERROR = "transport_error"


@dataclass(frozen=True, slots=True)
class ObjectFeatureShardSpec:
    shard_index: int
    sheet_index: int
    sheet_name: str
    slot_ids: tuple[str, ...]
    feature_ids: tuple[str, ...]
    packet_digest: str
    spec_digest: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.shard_index, bool)
            or not isinstance(self.shard_index, int)
            or self.shard_index < 0
            or isinstance(self.sheet_index, bool)
            or not isinstance(self.sheet_index, int)
            or self.sheet_index < 0
        ):
            raise PrototypeObjectProtocolError("shard indices must be nonnegative integers")
        if not isinstance(self.sheet_name, str) or not self.sheet_name:
            raise PrototypeObjectProtocolError("shard sheet name is invalid")
        if not 0 < len(self.slot_ids) <= MAX_FEATURE_SHARD_SLOTS:
            raise PrototypeObjectProtocolError("shard slot bound differs")
        if len(set(self.slot_ids)) != len(self.slot_ids):
            raise PrototypeObjectProtocolError("shard slots are duplicated")
        if not 0 < len(self.feature_ids) <= MAX_FEATURE_SHARD_FEATURES:
            raise PrototypeObjectProtocolError("shard feature bound differs")
        indices = tuple(OBJECT_FEATURE_IDS.index(item) for item in self.feature_ids)
        if indices != tuple(range(indices[0], indices[0] + len(indices))):
            raise PrototypeObjectProtocolError("shard features are not a catalog slice")
        if not _SHA256.fullmatch(self.packet_digest):
            raise PrototypeObjectProtocolError("shard packet digest is invalid")
        if self.spec_digest != canonical_digest(self.content_data()):
            raise PrototypeObjectProtocolError("shard digest differs")

    def content_data(self) -> dict[str, object]:
        return {
            "schema": "gkm.bongard-object-feature-shard-spec.v1",
            "shard_index": self.shard_index,
            "sheet_index": self.sheet_index,
            "sheet_name": self.sheet_name,
            "slot_ids": list(self.slot_ids),
            "feature_ids": list(self.feature_ids),
            "packet_digest": self.packet_digest,
            "profile_blind": True,
        }

    def to_data(self) -> dict[str, object]:
        return {**self.content_data(), "spec_digest": self.spec_digest}

    @classmethod
    def create(
        cls,
        *,
        shard_index: int,
        sheet_index: int,
        sheet_name: str,
        slot_ids: Sequence[str],
        feature_ids: Sequence[str],
        packet_digest: str,
    ) -> "ObjectFeatureShardSpec":
        provisional = {
            "schema": "gkm.bongard-object-feature-shard-spec.v1",
            "shard_index": shard_index,
            "sheet_index": sheet_index,
            "sheet_name": sheet_name,
            "slot_ids": list(slot_ids),
            "feature_ids": list(feature_ids),
            "packet_digest": packet_digest,
            "profile_blind": True,
        }
        return cls(
            shard_index,
            sheet_index,
            sheet_name,
            tuple(slot_ids),
            tuple(feature_ids),
            packet_digest,
            canonical_digest(provisional),
        )

    @classmethod
    def from_data(cls, value: object) -> "ObjectFeatureShardSpec":
        raw = _exact(
            value,
            {
                "schema", "shard_index", "sheet_index", "sheet_name",
                "slot_ids", "feature_ids", "packet_digest", "profile_blind",
                "spec_digest",
            },
            "feature shard spec",
        )
        if raw["schema"] != "gkm.bongard-object-feature-shard-spec.v1" or raw["profile_blind"] is not True:
            raise PrototypeObjectProtocolError("feature shard policy differs")
        return cls(
            raw["shard_index"],
            raw["sheet_index"],
            raw["sheet_name"],
            tuple(_rows(raw["slot_ids"], "shard slot IDs")),
            tuple(_rows(raw["feature_ids"], "shard feature IDs")),
            raw["packet_digest"],
            raw["spec_digest"],
        )


@dataclass(frozen=True, slots=True)
class ObjectFeatureShardPlan:
    packet_digest: str
    feature_catalog_digest: str
    shards: tuple[ObjectFeatureShardSpec, ...]
    plan_digest: str

    def content_data(self) -> dict[str, object]:
        return {
            "schema": "gkm.bongard-object-feature-shard-plan.v1",
            "packet_digest": self.packet_digest,
            "feature_catalog_digest": self.feature_catalog_digest,
            "max_slots": MAX_FEATURE_SHARD_SLOTS,
            "max_features": MAX_FEATURE_SHARD_FEATURES,
            "shards": [item.to_data() for item in self.shards],
            "order": "sheet-major-feature-slice-minor",
            "exact_duplicate_free_cover": True,
            "profile_blind": True,
        }

    def __post_init__(self) -> None:
        if not _SHA256.fullmatch(self.packet_digest):
            raise PrototypeObjectProtocolError("shard plan packet digest is invalid")
        if self.feature_catalog_digest != OBJECT_FEATURE_CATALOG_DIGEST:
            raise PrototypeObjectProtocolError("shard plan feature catalog differs")
        if tuple(item.shard_index for item in self.shards) != tuple(range(len(self.shards))):
            raise PrototypeObjectProtocolError("shard indices are not consecutive")
        if any(item.packet_digest != self.packet_digest for item in self.shards):
            raise PrototypeObjectProtocolError("shard belongs to another packet")
        covered = tuple(
            (slot_id, feature_id)
            for shard in self.shards
            for slot_id in shard.slot_ids
            for feature_id in shard.feature_ids
        )
        if len(covered) != len(set(covered)):
            raise PrototypeObjectProtocolError("shard plan coverage is duplicated")
        if self.plan_digest != canonical_digest(self.content_data()):
            raise PrototypeObjectProtocolError("shard plan digest differs")

    def to_data(self) -> dict[str, object]:
        return {**self.content_data(), "plan_digest": self.plan_digest}

    @classmethod
    def from_data(cls, value: object) -> "ObjectFeatureShardPlan":
        raw = _exact(
            value,
            {
                "schema", "packet_digest", "feature_catalog_digest", "max_slots",
                "max_features", "shards", "order", "exact_duplicate_free_cover",
                "profile_blind", "plan_digest",
            },
            "feature shard plan",
        )
        if (
            raw["schema"] != "gkm.bongard-object-feature-shard-plan.v1"
            or raw["max_slots"] != MAX_FEATURE_SHARD_SLOTS
            or raw["max_features"] != MAX_FEATURE_SHARD_FEATURES
            or raw["order"] != "sheet-major-feature-slice-minor"
            or raw["exact_duplicate_free_cover"] is not True
            or raw["profile_blind"] is not True
        ):
            raise PrototypeObjectProtocolError("feature shard plan policy differs")
        return cls(
            raw["packet_digest"],
            raw["feature_catalog_digest"],
            tuple(ObjectFeatureShardSpec.from_data(item) for item in _rows(raw["shards"], "feature shards")),
            raw["plan_digest"],
        )


@dataclass(frozen=True, slots=True)
class ParsedObjectFeatureShard:
    spec_digest: str
    audit_description: ObjectAuditText
    cells: tuple[ObjectFeatureCell, ...]
    payload_digest: str


@dataclass(frozen=True, slots=True)
class ObjectFeatureShardOutcome:
    spec_digest: str
    status: ObjectFeatureShardStatus
    cells: tuple[ObjectFeatureCell, ...]
    receipt_digest: str | None
    payload_digest: str | None
    reason_code: str | None
    error_type: str | None
    audit_description: ObjectAuditText

    def __post_init__(self) -> None:
        if not _SHA256.fullmatch(self.spec_digest):
            raise PrototypeObjectProtocolError("shard outcome spec digest is invalid")
        if not isinstance(self.status, ObjectFeatureShardStatus):
            raise TypeError("shard outcome status has the wrong type")
        if not isinstance(self.cells, tuple) or any(
            not isinstance(item, ObjectFeatureCell) for item in self.cells
        ):
            raise TypeError("shard outcome cells must be a typed tuple")
        if not isinstance(self.audit_description, ObjectAuditText):
            raise TypeError("shard outcome audit description has the wrong type")
        if self.status is ObjectFeatureShardStatus.SUCCESS:
            if (
                self.receipt_digest is None
                or self.payload_digest is None
                or self.reason_code is not None
                or self.error_type is not None
                or any(item.state is ObjectFeatureCellState.ERROR for item in self.cells)
            ):
                raise PrototypeObjectProtocolError("successful shard outcome differs")
        elif self.status is ObjectFeatureShardStatus.PARSER_ERROR:
            if (
                self.cells
                or self.receipt_digest is None
                or self.payload_digest is None
                or self.reason_code is None
                or self.error_type is None
            ):
                raise PrototypeObjectProtocolError("parser-error shard outcome differs")
        else:
            if (
                self.cells
                or self.reason_code is None
                or self.error_type is None
                or self.payload_digest is not None
                or self.receipt_digest is not None
            ):
                raise PrototypeObjectProtocolError("transport-error shard outcome differs")
        for value in (self.receipt_digest, self.payload_digest):
            if value is not None and not _SHA256.fullmatch(value):
                raise PrototypeObjectProtocolError("shard outcome digest is invalid")
        if self.reason_code is not None:
            _reason(self.reason_code)
        if self.error_type is not None:
            _reason(self.error_type)


def _feature_catalog_lines() -> str:
    values: list[str] = []
    for item in OBJECT_FEATURE_CATALOG:
        maximum = "unbounded" if item.maximum is None else str(item.maximum)
        values.append(
            f"- {item.feature_id}; unit={item.unit}; range=0..{maximum}; "
            f"meaning={item.operational_description}"
        )
    return "\n".join(values)


def prototype_object_description_prompt() -> str:
    return (
        "Inspect six reference images in two neutral groups of three. For each "
        "group, write one concise sentence describing the recurring visible "
        "appearance, then nominate one or more matching feature identifiers "
        "from the complete frozen measurement catalog below. Ignore pose, "
        "scale, location, and incidental stroke variation. Return group_0 then "
        "group_1. Emit prose and feature identifiers only: do not choose an "
        "operator, threshold, number, polarity, weight, negation, disjunction, "
        "executable text, or hidden role. Feature order is ignored and "
        "canonicalized. Python applies the one frozen operationalization after "
        "this turn; the model cannot tune it.\n\nFrozen measurement catalog:\n"
        + _feature_catalog_lines()
    )


def prototype_object_description_output_schema() -> dict[str, object]:
    row = {
        "type": "object",
        "properties": {
            "group_id": {"type": "string", "enum": list(PROTOTYPE_GROUP_IDS)},
            "rubric": {"type": "string"},
            "feature_ids": {
                "type": "array",
                "items": {"type": "string", "enum": list(OBJECT_FEATURE_IDS)},
            },
        },
        "required": ["group_id", "rubric", "feature_ids"],
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
            "grammar": "prose-plus-feature-id-nominations-only",
            "operationalization": {
                "authority": "fixed-python",
                "count": {
                    "operator": ObjectProfileOperator.AT_LEAST.value,
                    "target": DESCRIPTION_COUNT_TARGET,
                },
                "ppm": {
                    "operator": ObjectProfileOperator.AT_LEAST.value,
                    "target": DESCRIPTION_SUPPORT_TARGET_PPM,
                },
                "same_hypothesis_conjunction": True,
            },
            **_authority_data(),
        }
    )


def _fixed_atom(feature_id: str) -> ObjectProfileAtom:
    try:
        spec = next(item for item in OBJECT_FEATURE_CATALOG if item.feature_id == feature_id)
    except StopIteration as exc:
        raise PrototypeObjectProtocolError("feature nomination is outside catalog") from exc
    return ObjectProfileAtom(
        feature_id,
        ObjectProfileOperator.AT_LEAST,
        DESCRIPTION_SUPPORT_TARGET_PPM if spec.unit == "ppm" else DESCRIPTION_COUNT_TARGET,
    )


def parse_prototype_object_description_payload(
    payload: object,
) -> ParsedPrototypeObjectDescription:
    raw = _exact(payload, {"profiles"}, "description payload")
    values = _rows(raw["profiles"], "description profiles")
    if len(values) != 2:
        raise PrototypeObjectProtocolError("description must exhaust two groups")
    audits: list[ObjectAuditText] = []
    families: list[tuple[str, ...]] = []
    profiles: list[ObjectProfile] = []
    for index, (value, group_id) in enumerate(
        zip(values, PROTOTYPE_GROUP_IDS, strict=True)
    ):
        row = _exact(
            value,
            {"group_id", "rubric", "feature_ids"},
            f"feature family {index}",
        )
        if row["group_id"] != group_id:
            raise PrototypeObjectProtocolError("description group order differs")
        audits.append(ObjectAuditText.defined(_audit_prose(row["rubric"], "rubric")))
        raw_feature_ids = _rows(row["feature_ids"], "feature nominations")
        if not raw_feature_ids:
            raise PrototypeObjectProtocolError("feature nominations cannot be empty")
        if any(not isinstance(item, str) or item not in OBJECT_FEATURE_IDS for item in raw_feature_ids):
            raise PrototypeObjectProtocolError("feature nomination is outside catalog")
        if len(raw_feature_ids) != len(set(raw_feature_ids)):
            raise PrototypeObjectProtocolError("feature nominations are duplicated")
        feature_ids = tuple(sorted(raw_feature_ids, key=OBJECT_FEATURE_IDS.index))
        families.append(feature_ids)
        atoms = tuple(_fixed_atom(feature_id) for feature_id in feature_ids)
        profiles.append(ObjectProfile.create(group_id, atoms))
    return ParsedPrototypeObjectDescription(
        tuple(audits),  # type: ignore[arg-type]
        tuple(families),  # type: ignore[arg-type]
        tuple(profiles),  # type: ignore[arg-type]
    )


def plan_prototype_object_feature_shards(
    packet: ObjectHypothesisPacket,
) -> ObjectFeatureShardPlan:
    """Derive the only allowed sheet/feature partition from pixels alone."""

    if not isinstance(packet, ObjectHypothesisPacket):
        raise TypeError("packet must be ObjectHypothesisPacket")
    packet_digest = packet.digest()
    shards: list[ObjectFeatureShardSpec] = []
    for sheet in packet.atlas_sheets:
        if not sheet.slots:
            continue
        slot_ids = tuple(item.slot_id for item in sheet.slots)
        for start in range(0, len(OBJECT_FEATURE_IDS), MAX_FEATURE_SHARD_FEATURES):
            feature_ids = OBJECT_FEATURE_IDS[start : start + MAX_FEATURE_SHARD_FEATURES]
            shards.append(
                ObjectFeatureShardSpec.create(
                    shard_index=len(shards),
                    sheet_index=sheet.sheet_index,
                    sheet_name=sheet.name,
                    slot_ids=slot_ids,
                    feature_ids=feature_ids,
                    packet_digest=packet_digest,
                )
            )
    frozen = tuple(shards)
    content = {
        "schema": "gkm.bongard-object-feature-shard-plan.v1",
        "packet_digest": packet_digest,
        "feature_catalog_digest": OBJECT_FEATURE_CATALOG_DIGEST,
        "max_slots": MAX_FEATURE_SHARD_SLOTS,
        "max_features": MAX_FEATURE_SHARD_FEATURES,
        "shards": [item.to_data() for item in frozen],
        "order": "sheet-major-feature-slice-minor",
        "exact_duplicate_free_cover": True,
        "profile_blind": True,
    }
    result = ObjectFeatureShardPlan(
        packet_digest,
        OBJECT_FEATURE_CATALOG_DIGEST,
        frozen,
        canonical_digest(content),
    )
    verify_prototype_object_feature_shard_plan(result, packet)
    return result


def verify_prototype_object_feature_shard_plan(
    plan: ObjectFeatureShardPlan, packet: ObjectHypothesisPacket
) -> ObjectFeatureShardPlan:
    if not isinstance(plan, ObjectFeatureShardPlan):
        raise TypeError("plan must be ObjectFeatureShardPlan")
    if plan.packet_digest != packet.digest():
        raise PrototypeObjectProtocolError("feature shard plan belongs to another packet")
    expected_cover = tuple(
        (slot.slot_id, feature_id)
        for sheet in packet.atlas_sheets
        for slot in sheet.slots
        for feature_id in OBJECT_FEATURE_IDS
    )
    actual_cover = tuple(
        (slot_id, feature_id)
        for shard in plan.shards
        for slot_id in shard.slot_ids
        for feature_id in shard.feature_ids
    )
    if actual_cover != expected_cover or len(actual_cover) != len(set(actual_cover)):
        raise PrototypeObjectProtocolError("feature shard plan is not an exact ordered cover")
    expected_specs: list[tuple[int, str, tuple[str, ...], tuple[str, ...]]] = []
    for sheet in packet.atlas_sheets:
        if not sheet.slots:
            continue
        for start in range(0, len(OBJECT_FEATURE_IDS), MAX_FEATURE_SHARD_FEATURES):
            expected_specs.append(
                (
                    sheet.sheet_index,
                    sheet.name,
                    tuple(item.slot_id for item in sheet.slots),
                    OBJECT_FEATURE_IDS[start : start + MAX_FEATURE_SHARD_FEATURES],
                )
            )
    if tuple(
        (item.sheet_index, item.sheet_name, item.slot_ids, item.feature_ids)
        for item in plan.shards
    ) != tuple(expected_specs):
        raise PrototypeObjectProtocolError("feature shard partition differs")
    return plan


def _sheet_slots(
    packet: ObjectHypothesisPacket, shard: ObjectFeatureShardSpec
) -> tuple[object, ...]:
    plan = plan_prototype_object_feature_shards(packet)
    if shard.shard_index >= len(plan.shards) or plan.shards[shard.shard_index] != shard:
        raise PrototypeObjectProtocolError("feature shard is outside canonical plan")
    try:
        sheet = next(
            item for item in packet.atlas_sheets if item.sheet_index == shard.sheet_index
        )
    except StopIteration as exc:
        raise PrototypeObjectProtocolError("feature shard sheet is absent") from exc
    if sheet.name != shard.sheet_name or tuple(item.slot_id for item in sheet.slots) != shard.slot_ids:
        raise PrototypeObjectProtocolError("feature shard sheet binding differs")
    return tuple(sheet.slots)


def prototype_object_feature_shard_prompt(
    packet: ObjectHypothesisPacket, shard: ObjectFeatureShardSpec
) -> str:
    slots = _sheet_slots(packet, shard)
    rendered_slots = "\n".join(
        f"- row={slot.row_index}; column={slot.column_index}; slot_id={slot.slot_id}"
        for slot in slots
    )
    active = "\n".join(
        f"- {item.feature_id}; unit={item.unit}; range=0.."
        f"{MAX_OBSERVED_COUNT if item.maximum is None else item.maximum}; "
        f"meaning={item.operational_description}"
        for item in OBJECT_FEATURE_CATALOG
        if item.feature_id in shard.feature_ids
    )
    return (
        "Inspect this one opaque contact sheet. Every occupied grid cell is an "
        "independent visual hypothesis. Measure every listed slot against every "
        "listed feature, in exact row order and frozen feature order. Return one "
        "matrix row per slot. In states use s for a scored closed integer interval "
        "and i only when no honest interval can be given. For i use null in both "
        "numeric arrays. Do not omit rows or columns. No chosen description, rule, "
        "reference, group, experimental role, or evaluation status is available. The "
        "partition is fixed from the full catalog independently of any later "
        "decision. Return one neutral audit sentence.\n\nOccupied slots:\n"
        + rendered_slots
        + "\n\nFrozen feature slice:\n"
        + active
    )


def prototype_object_feature_prompt(packet: ObjectHypothesisPacket) -> str:
    """Compatibility wrapper for packets whose canonical plan has one shard."""

    plan = plan_prototype_object_feature_shards(packet)
    if len(plan.shards) != 1:
        raise PrototypeObjectProtocolError("packet requires feature shard prompts")
    return prototype_object_feature_shard_prompt(packet, plan.shards[0])


def prototype_object_feature_output_schema() -> dict[str, object]:
    nullable_integer = {"anyOf": [{"type": "integer"}, {"type": "null"}]}
    row = {
        "type": "object",
        "properties": {
            "slot_id": {"type": "string"},
            "states": {"type": "array", "items": {"type": "string", "enum": ["s", "i"]}},
            "lowers": {"type": "array", "items": nullable_integer},
            "uppers": {"type": "array", "items": nullable_integer},
        },
        "required": ["slot_id", "states", "lowers", "uppers"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "description": {"type": "string"},
            "rows": {"type": "array", "items": row},
        },
        "required": ["description", "rows"],
        "additionalProperties": False,
    }


def prototype_object_feature_shard_protocol_digest(
    packet: ObjectHypothesisPacket, shard: ObjectFeatureShardSpec
) -> str:
    prompt = prototype_object_feature_shard_prompt(packet, shard)
    schema = prototype_object_feature_output_schema()
    plan = plan_prototype_object_feature_shards(packet)
    return canonical_digest(
        {
            "schema": FEATURE_SCHEMA_ID,
            "protocol_id": FEATURE_PROTOCOL_ID,
            "protocol_family_digest": prototype_object_feature_protocol_family_digest(),
            "hypothesis_packet_digest": packet.digest(),
            "shard_plan_digest": plan.plan_digest,
            "shard_spec_digest": shard.spec_digest,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "output_schema_digest": canonical_digest(schema),
            "max_valid_payload_bytes": MAX_FEATURE_SHARD_PAYLOAD_BYTES,
            "profile_blind": True,
            "reference_blind": True,
            **_authority_data(),
        }
    )


def prototype_object_feature_protocol_digest(packet: ObjectHypothesisPacket) -> str:
    plan = plan_prototype_object_feature_shards(packet)
    return canonical_digest(
        {
            "schema": "gkm.bongard-object-feature-sharded-protocol.v2",
            "protocol_id": FEATURE_PROTOCOL_ID,
            "protocol_family_digest": prototype_object_feature_protocol_family_digest(),
            "hypothesis_packet_digest": packet.digest(),
            "shard_plan_digest": plan.plan_digest,
            "ordered_shard_protocol_digests": [
                prototype_object_feature_shard_protocol_digest(packet, item)
                for item in plan.shards
            ],
            "coverage": "exact-duplicate-free-slot-by-feature-cartesian-cover",
            "profile_blind": True,
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
            "max_slots_per_shard": MAX_FEATURE_SHARD_SLOTS,
            "max_features_per_shard": MAX_FEATURE_SHARD_FEATURES,
            "max_valid_payload_bytes": MAX_FEATURE_SHARD_PAYLOAD_BYTES,
            "matrix_columns": ["states", "lowers", "uppers"],
            "coverage": "exact-duplicate-free-slot-by-feature-cartesian-cover",
            **_authority_data(),
        }
    )


def _reason(value: object) -> str:
    if not isinstance(value, str) or _REASON.fullmatch(value) is None:
        raise PrototypeObjectProtocolError("indeterminate reason is not bounded")
    return value


def parse_prototype_object_feature_shard_payload(
    packet: ObjectHypothesisPacket,
    shard: ObjectFeatureShardSpec,
    payload: object,
) -> ParsedObjectFeatureShard:
    if not isinstance(packet, ObjectHypothesisPacket):
        raise TypeError("packet must be ObjectHypothesisPacket")
    slots = _sheet_slots(packet, shard)
    raw = _exact(payload, {"description", "rows"}, "feature shard payload")
    try:
        payload_bytes = canonical_json(dict(raw))
    except (TypeError, ValueError, UnicodeError) as exc:
        raise PrototypeObjectProtocolError("feature shard payload is not canonical JSON") from exc
    if len(payload_bytes) > MAX_FEATURE_SHARD_PAYLOAD_BYTES:
        raise PrototypeObjectProtocolError("feature shard payload exceeds byte bound")
    try:
        audit = ObjectAuditText.defined(_audit_prose(raw["description"], "scene description"))
    except (PrototypeObjectProtocolError, TypeError, ValueError):
        audit = ObjectAuditText.rejected()
    values = _rows(raw["rows"], "feature matrix rows")
    if len(values) != len(slots):
        raise PrototypeObjectProtocolError("feature shard row coverage differs")
    cells: list[ObjectFeatureCell] = []
    by_feature = {item.feature_id: item for item in OBJECT_FEATURE_CATALOG}
    for row_index, (value, slot) in enumerate(zip(values, slots, strict=True)):
        row = _exact(value, {"slot_id", "states", "lowers", "uppers"}, f"feature row {row_index}")
        if row["slot_id"] != slot.slot_id:
            raise PrototypeObjectProtocolError("feature shard row order differs")
        states = _rows(row["states"], "feature states")
        lowers = _rows(row["lowers"], "feature lowers")
        uppers = _rows(row["uppers"], "feature uppers")
        if not (len(states) == len(lowers) == len(uppers) == len(shard.feature_ids)):
            raise PrototypeObjectProtocolError("feature shard column coverage differs")
        for feature_id, state, lower, upper in zip(
            shard.feature_ids, states, lowers, uppers, strict=True
        ):
            if state == "s":
                try:
                    interval = IntegerInterval(lower, upper)
                except (TypeError, ValueError) as exc:
                    raise PrototypeObjectProtocolError("scored interval is invalid") from exc
                maximum = by_feature[feature_id].maximum
                if interval.upper > (MAX_OBSERVED_COUNT if maximum is None else maximum):
                    raise PrototypeObjectProtocolError("scored interval exceeds protocol range")
                cells.append(
                    ObjectFeatureCell(slot.hypothesis_id, feature_id, ObjectFeatureCellState.SCORED, interval)
                )
            elif state == "i":
                if lower is not None or upper is not None:
                    raise PrototypeObjectProtocolError("indeterminate matrix cell carries interval")
                cells.append(
                    ObjectFeatureCell(
                        slot.hypothesis_id,
                        feature_id,
                        ObjectFeatureCellState.INDETERMINATE,
                        None,
                        MODEL_INDETERMINATE_REASON,
                    )
                )
            else:
                raise PrototypeObjectProtocolError("unknown matrix state")
    return ParsedObjectFeatureShard(
        shard.spec_digest,
        audit,
        tuple(cells),
        canonical_digest(dict(raw)),
    )


def _error_cells_for_shard(
    packet: ObjectHypothesisPacket,
    shard: ObjectFeatureShardSpec,
    *,
    reason_code: str,
    error_type: str,
) -> tuple[ObjectFeatureCell, ...]:
    slots = _sheet_slots(packet, shard)
    return tuple(
        ObjectFeatureCell(
            slot.hypothesis_id,
            feature_id,
            ObjectFeatureCellState.ERROR,
            None,
            _reason(reason_code),
            _reason(error_type),
        )
        for slot in slots
        for feature_id in shard.feature_ids
    )


def assemble_prototype_object_feature_shards(
    packet: ObjectHypothesisPacket,
    plan: ObjectFeatureShardPlan,
    outcomes: Sequence[ObjectFeatureShardOutcome],
    *,
    feature_model_id: str,
) -> ParsedPrototypeObjectFeatures:
    verify_prototype_object_feature_shard_plan(plan, packet)
    frozen = tuple(outcomes)
    if len(frozen) != len(plan.shards) or tuple(item.spec_digest for item in frozen) != tuple(
        item.spec_digest for item in plan.shards
    ):
        raise PrototypeObjectProtocolError("feature shard outcomes do not exactly cover plan")
    parsed_by_key: dict[tuple[str, str], ObjectFeatureCell] = {}
    audits: list[ObjectAuditText] = []
    for shard, outcome in zip(plan.shards, frozen, strict=True):
        audits.append(outcome.audit_description)
        cells = outcome.cells if outcome.status is ObjectFeatureShardStatus.SUCCESS else _error_cells_for_shard(
            packet,
            shard,
            reason_code=outcome.reason_code or "shard_failed",
            error_type=outcome.error_type or "ObjectFeatureShardFailure",
        )
        expected_keys = tuple(
            (slot_id, feature_id)
            for slot_id in shard.slot_ids
            for feature_id in shard.feature_ids
        )
        expected_slots = _sheet_slots(packet, shard)
        expected_cells = tuple(
            (slot, feature_id)
            for slot in expected_slots
            for feature_id in shard.feature_ids
        )
        if len(cells) != len(expected_cells) or any(
            cell.hypothesis_id != slot.hypothesis_id or cell.feature_id != feature_id
            for cell, (slot, feature_id) in zip(cells, expected_cells, strict=True)
        ):
            raise PrototypeObjectProtocolError("feature shard outcome cell coverage differs")
        for key, cell in zip(expected_keys, cells, strict=True):
            if key in parsed_by_key:
                raise PrototypeObjectProtocolError("feature shard outcome duplicates a cell")
            parsed_by_key[key] = cell

    expected_all = tuple(
        (slot.slot_id, feature_id)
        for sheet in packet.atlas_sheets
        for slot in sheet.slots
        for feature_id in OBJECT_FEATURE_IDS
    )
    if tuple(parsed_by_key) != expected_all:
        raise PrototypeObjectProtocolError("assembled feature cells are not exhaustive")
    hypothesis_digest = packet.digest()
    protocol_digest = prototype_object_feature_protocol_digest(packet)
    receipt_manifest_digest = canonical_digest(
        {
            "schema": "gkm.bongard-object-feature-receipt-manifest.v1",
            "plan_digest": plan.plan_digest,
            "ordered": [
                {"status": item.status.value, "receipt_digest": item.receipt_digest}
                for item in frozen
            ],
        }
    )
    payload_manifest_digest = canonical_digest(
        {
            "schema": "gkm.bongard-object-feature-payload-manifest.v1",
            "plan_digest": plan.plan_digest,
            "ordered": [
                {"status": item.status.value, "payload_digest": item.payload_digest}
                for item in frozen
            ],
        }
    )
    slot_by_hypothesis = {
        (slot.scenario_id, slot.hypothesis_id): slot
        for sheet in packet.atlas_sheets
        for slot in sheet.slots
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
            cells.extend(
                parsed_by_key[(slot.slot_id, feature_id)]
                for feature_id in OBJECT_FEATURE_IDS
            )
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
                feature_receipt_digest=receipt_manifest_digest,
                feature_payload_digest=payload_manifest_digest,
            )
        )
    audit = next(
        (item for item in audits if item.state is ObjectAuditTextState.DEFINED),
        ObjectAuditText.rejected(),
    )
    return ParsedPrototypeObjectFeatures(audit, tuple(packets))


def parse_prototype_object_feature_payload(
    packet: ObjectHypothesisPacket,
    payload: object,
    *,
    feature_model_id: str,
    feature_receipt_digest: str,
) -> ParsedPrototypeObjectFeatures:
    """Compatibility entry point for a canonical one-shard packet."""

    plan = plan_prototype_object_feature_shards(packet)
    if len(plan.shards) != 1:
        raise PrototypeObjectProtocolError("packet requires exhaustive shard assembly")
    parsed = parse_prototype_object_feature_shard_payload(packet, plan.shards[0], payload)
    outcome = ObjectFeatureShardOutcome(
        parsed.spec_digest,
        ObjectFeatureShardStatus.SUCCESS,
        parsed.cells,
        feature_receipt_digest,
        parsed.payload_digest,
        None,
        None,
        parsed.audit_description,
    )
    return assemble_prototype_object_feature_shards(
        packet, plan, (outcome,), feature_model_id=feature_model_id
    )


__all__ = (
    "DESCRIPTION_COUNT_TARGET",
    "DESCRIPTION_PROTOCOL_ID",
    "DESCRIPTION_SUPPORT_TARGET_PPM",
    "FEATURE_PROTOCOL_ID",
    "MAX_FEATURE_SHARD_CELLS",
    "MAX_FEATURE_SHARD_FEATURES",
    "MAX_FEATURE_SHARD_PAYLOAD_BYTES",
    "MAX_FEATURE_SHARD_SLOTS",
    "MODEL_INDETERMINATE_REASON",
    "ObjectAuditText",
    "ObjectAuditTextState",
    "ObjectFeatureShardOutcome",
    "ObjectFeatureShardPlan",
    "ObjectFeatureShardSpec",
    "ObjectFeatureShardStatus",
    "ParsedObjectFeatureShard",
    "ParsedPrototypeObjectDescription",
    "ParsedPrototypeObjectFeatures",
    "PrototypeObjectProtocolError",
    "assemble_prototype_object_feature_shards",
    "parse_prototype_object_description_payload",
    "parse_prototype_object_feature_payload",
    "parse_prototype_object_feature_shard_payload",
    "plan_prototype_object_feature_shards",
    "prototype_object_description_output_schema",
    "prototype_object_description_prompt",
    "prototype_object_description_protocol_digest",
    "prototype_object_feature_output_schema",
    "prototype_object_feature_prompt",
    "prototype_object_feature_protocol_family_digest",
    "prototype_object_feature_protocol_digest",
    "prototype_object_feature_shard_prompt",
    "prototype_object_feature_shard_protocol_digest",
    "prototype_object_protocol_source_digest",
    "verify_prototype_object_feature_shard_plan",
)

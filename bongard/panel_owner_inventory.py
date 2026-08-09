"""Candidate-independent raw-panel owner inventory for typed visual predicates.

One neutral ``panel.png`` is mapped to a fixed twelve-slot response.  Python
validates the closed sentinels, rejects malformed scalar types, and assigns
canonical ``OwnerId`` values from kind, Grid16 geometry, and canonical parent
structure.  Model slot names are temporary references and never become owner
identity.

This additive module is transport-agnostic and is not connected to the live
panel-soft v2 runner.  Its receipt values are custody commitments for a
trusted verifier process, not a sandbox against hostile in-process Python.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import re
from typing import Any, Mapping, Protocol

from bongard.canonical import canonical_digest
from bongard.panel_soft_ontology import (
    EnumerationResolution,
    OwnerId,
    OwnerInventory,
    OwnerKind,
    PanelLocalOwner,
    QuantizedPoint,
    QuantizedRegion,
)
from bongard.transport import MAX_PANEL_PNG_BYTES, validate_codex_strict_output_schema


PANEL_OWNER_SLOT_CAPACITY = 12
PANEL_OWNER_SLOT_NAMES = tuple(
    f"slot_{index:02d}" for index in range(PANEL_OWNER_SLOT_CAPACITY)
)
PANEL_OWNER_NEUTRAL_IMAGE_NAME = "panel.png"

PANEL_OWNER_SLOT_SCHEMA = "gkm.bongard-panel-owner-raw-slot.v1"
PANEL_OWNER_PARSED_SCHEMA = "gkm.bongard-panel-owner-parsed-response.v1"
PANEL_OWNER_GAP_SCHEMA = "gkm.bongard-panel-owner-inventory-gap.v1"
PANEL_OWNER_RECEIPT_SCHEMA = "gkm.bongard-panel-owner-call-receipt.v1"
PANEL_OWNER_ARTIFACT_SCHEMA = "gkm.bongard-panel-owner-inventory-artifact.v1"
PANEL_OWNER_SEMANTIC_SCHEMA = "gkm.bongard-panel-owner-inventory-semantic.v1"
PANEL_OWNER_CONTRACT_SCHEMA = "gkm.bongard-panel-owner-inventory-contract.v1"
PANEL_OWNER_REQUEST_SCHEMA = "gkm.bongard-panel-owner-inventory-request.v1"
PANEL_OWNER_SEMANTICS_SCHEMA = "gkm.bongard-panel-owner-semantics.v1"
PANEL_OWNER_ALGORITHM_ID = (
    "bongard.panel-owner-inventory/grid16-parent-path-canonicalization-v1"
)

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_CODE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
_TEMP_SLOT = re.compile(r"slot_[0-9]{2}\Z")


class PanelOwnerInventoryError(ValueError):
    """A raw response, canonical inventory, receipt, or artifact is invalid."""


class InventoryStatus(str, Enum):
    COMPLETE = "complete"
    UNRESOLVED = "unresolved"
    CAPACITY_EXCEEDED = "capacity_exceeded"
    ERROR = "error"


class RawSlotState(str, Enum):
    OWNER = "owner"
    UNUSED = "unused"
    UNRESOLVED = "unresolved"
    CAPACITY_EXCEEDED = "capacity_exceeded"
    ERROR = "error"


class InventoryGapKind(str, Enum):
    UNRESOLVED = "unresolved"
    CAPACITY_EXCEEDED = "capacity_exceeded"
    SEMANTIC_COLLISION = "semantic_collision"
    PARENT_CYCLE = "parent_cycle"
    INVALID_PARENT = "invalid_parent"
    TRANSPORT_ERROR = "transport_error"


class InventoryTransportKind(str, Enum):
    CODEX_NAMED_IMAGE = "codex_named_image"
    INJECTED_RECEIPTED = "injected_receipted"


_OWNER_KIND_VALUES = tuple(sorted(item.value for item in OwnerKind))
_OWNER_KIND_OR_SENTINEL = (*_OWNER_KIND_VALUES, "not_applicable")
_PARENT_VALUES = (*PANEL_OWNER_SLOT_NAMES, "root", "not_applicable")


def _fields(value: object, expected: set[str], label: str) -> Mapping[str, Any]:
    if (
        not isinstance(value, Mapping)
        or any(type(key) is not str for key in value)
        or set(value) != expected
    ):
        raise PanelOwnerInventoryError(f"{label} fields differ")
    return value


def _digest(value: object, label: str) -> str:
    if type(value) is not str or _DIGEST.fullmatch(value) is None:
        raise PanelOwnerInventoryError(f"{label} must be a lowercase SHA-256")
    return value


def _code(value: object, label: str) -> str:
    if (
        type(value) is not str
        or _CODE.fullmatch(value) is None
        or "lean" in value.casefold()
    ):
        raise PanelOwnerInventoryError(f"{label} must be a bounded safe code")
    return value


def _exact_int(value: object, label: str) -> int:
    if type(value) is not int:
        raise PanelOwnerInventoryError(f"{label} must be an exact integer")
    return value


def _canonical_roundtrip(value: object, raw: Mapping[str, Any], label: str) -> None:
    if value.to_data() != dict(raw):
        raise PanelOwnerInventoryError(f"{label} is not canonical")


def panel_owner_inventory_prompt() -> str:
    """Return the sole role-blind text shown with one neutral panel image."""

    return (
        "Inspect panel.png by itself. Enumerate the visually coherent owners "
        "that are directly visible. An owner is one coherent figure, trace, "
        "loop, segment, or marker. Use Grid16 coordinates from 0 through 15 "
        "for an inclusive bounding box. A parent is a directly containing or "
        "structurally governing owner; otherwise use root. Temporary slot "
        "labels exist only so a child can refer to a parent.\n\n"
        "Return complete only when every visible owner fits in twelve slots "
        "and every parent and box is resolved. Fill remaining slots with the "
        "unused sentinel. Return unresolved with the unresolved sentinel in "
        "all slots when ownership, a parent, or a box cannot be resolved. "
        "Return capacity_exceeded with that sentinel in all slots when more "
        "than twelve owners are visible. Do not guess through ambiguity."
    )


def _slot_output_schema(*, model_visible: bool) -> dict[str, object]:
    states = [
        RawSlotState.OWNER.value,
        RawSlotState.UNUSED.value,
        RawSlotState.UNRESOLVED.value,
        RawSlotState.CAPACITY_EXCEEDED.value,
    ]
    if not model_visible:
        states.append(RawSlotState.ERROR.value)
    return {
        "type": "object",
        "properties": {
            "slot_state": {"type": "string", "enum": states},
            "owner_kind": {
                "type": "string",
                "enum": list(_OWNER_KIND_OR_SENTINEL),
            },
            "parent_slot": {"type": "string", "enum": list(_PARENT_VALUES)},
            "x_min": {"type": "integer"},
            "y_min": {"type": "integer"},
            "x_max": {"type": "integer"},
            "y_max": {"type": "integer"},
        },
        "required": [
            "slot_state",
            "owner_kind",
            "parent_slot",
            "x_min",
            "y_min",
            "x_max",
            "y_max",
        ],
        "additionalProperties": False,
    }


def panel_owner_inventory_output_schema() -> dict[str, object]:
    """Strict fixed-capacity schema for the model-visible response."""

    schema: dict[str, object] = {
        "type": "object",
        "properties": {
            "inventory_status": {
                "type": "string",
                "enum": [
                    InventoryStatus.COMPLETE.value,
                    InventoryStatus.UNRESOLVED.value,
                    InventoryStatus.CAPACITY_EXCEEDED.value,
                ],
            },
            "slots": {
                "type": "object",
                "properties": {
                    name: _slot_output_schema(model_visible=True)
                    for name in PANEL_OWNER_SLOT_NAMES
                },
                "required": list(PANEL_OWNER_SLOT_NAMES),
                "additionalProperties": False,
            },
        },
        "required": ["inventory_status", "slots"],
        "additionalProperties": False,
    }
    validate_codex_strict_output_schema(schema)
    return schema


def panel_owner_inventory_model_view() -> dict[str, object]:
    """The complete non-pixel model-visible view; it contains no experiment role."""

    return {
        "image_name": PANEL_OWNER_NEUTRAL_IMAGE_NAME,
        "prompt": panel_owner_inventory_prompt(),
        "output_schema": panel_owner_inventory_output_schema(),
    }


def panel_owner_semantics_data() -> dict[str, object]:
    """Owner vocabulary identity, independent of every feature/spec catalog."""

    return {
        "schema": PANEL_OWNER_SEMANTICS_SCHEMA,
        "owner_kinds": sorted(item.value for item in OwnerKind),
        "grid_unit": "grid16_inclusive_bbox",
        "grid_bin_count": 16,
        "parent_arity": "zero_or_one",
        "owner_id_rule": "semantic_kind_geometry_parent_path",
    }


def panel_owner_semantics_digest() -> str:
    return canonical_digest(panel_owner_semantics_data())


def panel_owner_inventory_contract_data() -> dict[str, object]:
    return {
        "schema": PANEL_OWNER_CONTRACT_SCHEMA,
        "algorithm_id": PANEL_OWNER_ALGORITHM_ID,
        "owner_semantics_digest": panel_owner_semantics_digest(),
        "slot_capacity": PANEL_OWNER_SLOT_CAPACITY,
        "slot_names": list(PANEL_OWNER_SLOT_NAMES),
        "grid_unit": "grid16_inclusive_bbox",
        "neutral_image_name": PANEL_OWNER_NEUTRAL_IMAGE_NAME,
        "prompt_digest": canonical_digest(panel_owner_inventory_prompt()),
        "output_schema_digest": canonical_digest(
            panel_owner_inventory_output_schema()
        ),
        "owner_id_source": "semantic_kind_geometry_parent_path",
        "model_slot_order_used_for_identity": False,
    }


def panel_owner_inventory_contract_digest() -> str:
    return canonical_digest(panel_owner_inventory_contract_data())


@dataclass(frozen=True, slots=True)
class RawInventorySlot:
    temporary_name: str
    state: RawSlotState
    owner_kind: OwnerKind | None
    parent_slot: str | None
    bbox: QuantizedRegion | None

    def __post_init__(self) -> None:
        if (
            type(self.temporary_name) is not str
            or self.temporary_name not in PANEL_OWNER_SLOT_NAMES
        ):
            raise PanelOwnerInventoryError("temporary slot name differs")
        if type(self.state) is not RawSlotState:
            raise TypeError("raw slot state must be RawSlotState")
        if self.state is RawSlotState.OWNER:
            if type(self.owner_kind) is not OwnerKind:
                raise TypeError("owner slot needs OwnerKind")
            if self.parent_slot is not None and self.parent_slot not in PANEL_OWNER_SLOT_NAMES:
                raise PanelOwnerInventoryError("owner parent slot differs")
            if type(self.bbox) is not QuantizedRegion:
                raise TypeError("owner slot needs a Grid16 bbox")
        elif self.owner_kind is not None or self.parent_slot is not None or self.bbox is not None:
            raise PanelOwnerInventoryError("sentinel slot carries owner fields")

    def to_data(self) -> dict[str, object]:
        if self.state is RawSlotState.OWNER:
            assert self.owner_kind is not None and self.bbox is not None
            return {
                "slot_state": self.state.value,
                "owner_kind": self.owner_kind.value,
                "parent_slot": "root" if self.parent_slot is None else self.parent_slot,
                "x_min": self.bbox.minimum.x,
                "y_min": self.bbox.minimum.y,
                "x_max": self.bbox.maximum.x,
                "y_max": self.bbox.maximum.y,
            }
        return {
            "slot_state": self.state.value,
            "owner_kind": "not_applicable",
            "parent_slot": "not_applicable",
            "x_min": -1,
            "y_min": -1,
            "x_max": -1,
            "y_max": -1,
        }

    @classmethod
    def from_data(cls, temporary_name: str, value: object) -> "RawInventorySlot":
        raw = _fields(
            value,
            {
                "slot_state",
                "owner_kind",
                "parent_slot",
                "x_min",
                "y_min",
                "x_max",
                "y_max",
            },
            f"raw inventory {temporary_name}",
        )
        try:
            state = RawSlotState(raw["slot_state"])
        except (TypeError, ValueError) as exc:
            raise PanelOwnerInventoryError("raw slot state differs") from exc
        coordinates = tuple(
            _exact_int(raw[name], f"{temporary_name}.{name}")
            for name in ("x_min", "y_min", "x_max", "y_max")
        )
        if state is RawSlotState.OWNER:
            try:
                kind = OwnerKind(raw["owner_kind"])
            except (TypeError, ValueError) as exc:
                raise PanelOwnerInventoryError("raw owner kind differs") from exc
            parent = raw["parent_slot"]
            if type(parent) is not str or parent not in (*PANEL_OWNER_SLOT_NAMES, "root"):
                raise PanelOwnerInventoryError("raw owner parent differs")
            if any(not 0 <= item < 16 for item in coordinates):
                raise PanelOwnerInventoryError("raw owner bbox is outside Grid16")
            try:
                bbox = QuantizedRegion(
                    QuantizedPoint(coordinates[0], coordinates[1]),
                    QuantizedPoint(coordinates[2], coordinates[3]),
                )
            except (TypeError, ValueError) as exc:
                raise PanelOwnerInventoryError("raw owner bbox differs") from exc
            result = cls(
                temporary_name,
                state,
                kind,
                None if parent == "root" else parent,
                bbox,
            )
        else:
            if (
                raw["owner_kind"] != "not_applicable"
                or raw["parent_slot"] != "not_applicable"
                or coordinates != (-1, -1, -1, -1)
            ):
                raise PanelOwnerInventoryError("raw slot sentinel differs")
            result = cls(temporary_name, state, None, None, None)
        if result.to_data() != dict(raw):
            raise PanelOwnerInventoryError("raw inventory slot is not canonical")
        return result


@dataclass(frozen=True, slots=True)
class ParsedPanelOwnerResponse:
    status: InventoryStatus
    slots: tuple[RawInventorySlot, ...]

    def __post_init__(self) -> None:
        if type(self.status) is not InventoryStatus:
            raise TypeError("parsed inventory status must be InventoryStatus")
        if type(self.slots) is not tuple or len(self.slots) != PANEL_OWNER_SLOT_CAPACITY:
            raise PanelOwnerInventoryError("parsed inventory must have twelve slots")
        if any(type(item) is not RawInventorySlot for item in self.slots):
            raise TypeError("parsed inventory contains a non-slot")
        names = tuple(item.temporary_name for item in self.slots)
        if names != PANEL_OWNER_SLOT_NAMES:
            raise PanelOwnerInventoryError("parsed inventory slots are not canonical")
        allowed = {
            InventoryStatus.COMPLETE: {RawSlotState.OWNER, RawSlotState.UNUSED},
            InventoryStatus.UNRESOLVED: {RawSlotState.UNRESOLVED},
            InventoryStatus.CAPACITY_EXCEEDED: {RawSlotState.CAPACITY_EXCEEDED},
            InventoryStatus.ERROR: {RawSlotState.ERROR},
        }[self.status]
        states = {item.state for item in self.slots}
        if not states or not states <= allowed:
            raise PanelOwnerInventoryError("inventory status and slot sentinels differ")
        if self.status is not InventoryStatus.COMPLETE and states != allowed:
            raise PanelOwnerInventoryError("noncomplete response must use one sentinel")

    @property
    def owner_slots(self) -> tuple[RawInventorySlot, ...]:
        return tuple(item for item in self.slots if item.state is RawSlotState.OWNER)

    def to_data(self) -> dict[str, object]:
        return {
            "inventory_status": self.status.value,
            "slots": {item.temporary_name: item.to_data() for item in self.slots},
        }


def parse_panel_owner_inventory_payload(value: object) -> ParsedPanelOwnerResponse:
    raw = _fields(value, {"inventory_status", "slots"}, "panel owner response")
    try:
        status = InventoryStatus(raw["inventory_status"])
    except (TypeError, ValueError) as exc:
        raise PanelOwnerInventoryError("panel owner response status differs") from exc
    slots = _fields(raw["slots"], set(PANEL_OWNER_SLOT_NAMES), "panel owner slots")
    result = ParsedPanelOwnerResponse(
        status,
        tuple(RawInventorySlot.from_data(name, slots[name]) for name in PANEL_OWNER_SLOT_NAMES),
    )
    if result.to_data() != dict(raw):
        raise PanelOwnerInventoryError("panel owner response is not canonical")
    return result


@dataclass(frozen=True, order=True, slots=True)
class InventoryGap:
    kind: InventoryGapKind
    evidence_digest: str

    def __post_init__(self) -> None:
        if type(self.kind) is not InventoryGapKind:
            raise TypeError("inventory gap kind must be InventoryGapKind")
        _digest(self.evidence_digest, "inventory gap evidence digest")

    @property
    def gap_digest(self) -> str:
        return canonical_digest(self.to_data())

    def to_data(self) -> dict[str, object]:
        return {
            "schema": PANEL_OWNER_GAP_SCHEMA,
            "kind": self.kind.value,
            "evidence_digest": self.evidence_digest,
        }

    @classmethod
    def from_data(cls, value: object) -> "InventoryGap":
        raw = _fields(value, {"schema", "kind", "evidence_digest"}, "inventory gap")
        if raw["schema"] != PANEL_OWNER_GAP_SCHEMA:
            raise PanelOwnerInventoryError("inventory gap schema differs")
        try:
            result = cls(
                InventoryGapKind(raw["kind"]),
                raw["evidence_digest"],
            )
        except (TypeError, ValueError) as exc:
            if isinstance(exc, PanelOwnerInventoryError):
                raise
            raise PanelOwnerInventoryError("inventory gap value differs") from exc
        _canonical_roundtrip(result, raw, "inventory gap")
        return result


@dataclass(frozen=True, slots=True)
class CanonicalOwnerInventory:
    status: InventoryStatus
    owners: tuple[PanelLocalOwner, ...]
    gap: InventoryGap | None

    def __post_init__(self) -> None:
        if type(self.status) is not InventoryStatus:
            raise TypeError("canonical inventory status must be InventoryStatus")
        if type(self.owners) is not tuple or any(
            type(item) is not PanelLocalOwner for item in self.owners
        ):
            raise TypeError("canonical owners must be a PanelLocalOwner tuple")
        owner_names = tuple(item.owner_id.value for item in self.owners)
        if owner_names != tuple(sorted(owner_names)) or len(owner_names) != len(
            set(owner_names)
        ):
            raise PanelOwnerInventoryError("canonical owners must be unique and sorted")
        if self.status is InventoryStatus.COMPLETE:
            if self.gap is not None:
                raise PanelOwnerInventoryError("complete inventory cannot carry a gap")
        elif self.owners or type(self.gap) is not InventoryGap:
            raise PanelOwnerInventoryError("noncomplete inventory needs one gap and no owners")


def _local_owner_data(slot: RawInventorySlot) -> dict[str, object]:
    if slot.state is not RawSlotState.OWNER:
        raise TypeError("semantic owner data requires an owner slot")
    assert slot.owner_kind is not None and slot.bbox is not None
    return {
        "kind": slot.owner_kind.value,
        "bbox": {
            "x_min": slot.bbox.minimum.x,
            "y_min": slot.bbox.minimum.y,
            "x_max": slot.bbox.maximum.x,
            "y_max": slot.bbox.maximum.y,
        },
    }


def _local_owner_key(slot: RawInventorySlot) -> tuple[object, ...]:
    assert slot.owner_kind is not None and slot.bbox is not None
    return (
        slot.owner_kind.value,
        slot.bbox.minimum.x,
        slot.bbox.minimum.y,
        slot.bbox.maximum.x,
        slot.bbox.maximum.y,
    )


def _typed_gap(kind: InventoryGapKind, evidence: object) -> CanonicalOwnerInventory:
    return CanonicalOwnerInventory(
        InventoryStatus.CAPACITY_EXCEEDED
        if kind is InventoryGapKind.CAPACITY_EXCEEDED
        else InventoryStatus.ERROR
        if kind is InventoryGapKind.TRANSPORT_ERROR
        else InventoryStatus.UNRESOLVED,
        (),
        InventoryGap(
            kind,
            canonical_digest(
                {
                    "schema": "gkm.bongard-panel-owner-gap-evidence.v1",
                    "kind": kind.value,
                    "evidence": evidence,
                }
            ),
        ),
    )


def canonicalize_panel_owner_response(
    parsed: ParsedPanelOwnerResponse,
) -> CanonicalOwnerInventory:
    """Assign owner IDs from semantics and parent paths, never temporary slots."""

    if type(parsed) is not ParsedPanelOwnerResponse:
        raise TypeError("canonicalization requires ParsedPanelOwnerResponse")
    if parsed.status is InventoryStatus.UNRESOLVED:
        return _typed_gap(
            InventoryGapKind.UNRESOLVED,
            {"sentinel": RawSlotState.UNRESOLVED.value},
        )
    if parsed.status is InventoryStatus.CAPACITY_EXCEEDED:
        return _typed_gap(
            InventoryGapKind.CAPACITY_EXCEEDED,
            {
                "slot_capacity": PANEL_OWNER_SLOT_CAPACITY,
                "sentinel": RawSlotState.CAPACITY_EXCEEDED.value,
            },
        )
    if parsed.status is InventoryStatus.ERROR:
        return _typed_gap(
            InventoryGapKind.TRANSPORT_ERROR,
            {"sentinel": RawSlotState.ERROR.value},
        )

    owner_by_slot = {item.temporary_name: item for item in parsed.owner_slots}
    invalid_children = tuple(
        sorted(
            (
                _local_owner_data(item)
                for item in parsed.owner_slots
                if item.parent_slot is not None
                and item.parent_slot not in owner_by_slot
            ),
            key=canonical_digest,
        )
    )
    if invalid_children:
        return _typed_gap(
            InventoryGapKind.INVALID_PARENT,
            {"children": list(invalid_children)},
        )

    visit_state: dict[str, int] = {}
    stack: list[str] = []
    cycle_rows: list[dict[str, object]] | None = None

    def visit(slot_name: str) -> None:
        nonlocal cycle_rows
        state = visit_state.get(slot_name, 0)
        if state == 2 or cycle_rows is not None:
            return
        if state == 1:
            start = stack.index(slot_name)
            cycle_rows = sorted(
                (_local_owner_data(owner_by_slot[name]) for name in stack[start:]),
                key=canonical_digest,
            )
            return
        visit_state[slot_name] = 1
        stack.append(slot_name)
        parent = owner_by_slot[slot_name].parent_slot
        if parent is not None:
            visit(parent)
        stack.pop()
        visit_state[slot_name] = 2

    for name in sorted(owner_by_slot):
        visit(name)
    if cycle_rows is not None:
        return _typed_gap(
            InventoryGapKind.PARENT_CYCLE,
            {"cycle_rows": cycle_rows},
        )

    paths: dict[str, tuple[tuple[object, ...], ...]] = {}

    def semantic_path(slot_name: str) -> tuple[tuple[object, ...], ...]:
        retained = paths.get(slot_name)
        if retained is not None:
            return retained
        slot = owner_by_slot[slot_name]
        prefix = () if slot.parent_slot is None else semantic_path(slot.parent_slot)
        result = prefix + (_local_owner_key(slot),)
        paths[slot_name] = result
        return result

    for name in owner_by_slot:
        semantic_path(name)
    grouped: dict[tuple[tuple[object, ...], ...], list[str]] = {}
    for name, path in paths.items():
        grouped.setdefault(path, []).append(name)
    collisions = [path for path, names in grouped.items() if len(names) > 1]
    if collisions:
        collision_data = sorted(
            (
                [
                    {
                        "kind": row[0],
                        "bbox": list(row[1:]),
                    }
                    for row in path
                ]
                for path in collisions
            ),
            key=canonical_digest,
        )
        return _typed_gap(
            InventoryGapKind.SEMANTIC_COLLISION,
            {"semantic_paths": collision_data},
        )

    ordered_slots = tuple(sorted(owner_by_slot, key=lambda name: paths[name]))
    canonical_id = {
        name: OwnerId(f"owner_{index:04d}")
        for index, name in enumerate(ordered_slots, start=1)
    }
    owners = []
    for name in ordered_slots:
        raw = owner_by_slot[name]
        assert raw.owner_kind is not None and raw.bbox is not None
        owners.append(
            PanelLocalOwner(
                canonical_id[name],
                raw.owner_kind,
                raw.bbox,
                ()
                if raw.parent_slot is None
                else (canonical_id[raw.parent_slot],),
            )
        )
    return CanonicalOwnerInventory(InventoryStatus.COMPLETE, tuple(owners), None)


def _panel_png_identity(panel_png: object) -> tuple[bytes, str, int]:
    if type(panel_png) is not bytes:
        raise TypeError("panel PNG must be exact bytes")
    if not 0 < len(panel_png) <= MAX_PANEL_PNG_BYTES:
        raise PanelOwnerInventoryError("panel PNG byte count differs")
    if not panel_png.startswith(_PNG_SIGNATURE):
        raise PanelOwnerInventoryError("panel input is not a PNG")
    return panel_png, hashlib.sha256(panel_png).hexdigest(), len(panel_png)


def panel_owner_inventory_request_data(
    *,
    transport_kind: InventoryTransportKind,
    model_id: str,
    panel_png_digest: str,
    panel_png_byte_count: int,
    observer_contract_digest: str,
) -> dict[str, object]:
    if type(transport_kind) is not InventoryTransportKind:
        raise TypeError("inventory transport kind has the wrong type")
    _code(model_id, "inventory model ID")
    _digest(panel_png_digest, "inventory panel digest")
    if type(panel_png_byte_count) is not int or panel_png_byte_count <= 0:
        raise PanelOwnerInventoryError("panel byte count must be a positive exact int")
    _digest(observer_contract_digest, "inventory observer contract digest")
    return {
        "schema": PANEL_OWNER_REQUEST_SCHEMA,
        "transport_kind": transport_kind.value,
        "model_id": model_id,
        "image_name": PANEL_OWNER_NEUTRAL_IMAGE_NAME,
        "panel_png_digest": panel_png_digest,
        "panel_png_byte_count": panel_png_byte_count,
        "observer_contract_digest": observer_contract_digest,
        "prompt_digest": canonical_digest(panel_owner_inventory_prompt()),
        "output_schema_digest": canonical_digest(
            panel_owner_inventory_output_schema()
        ),
        "inventory_contract_digest": panel_owner_inventory_contract_digest(),
    }


@dataclass(frozen=True, slots=True)
class PanelOwnerInventoryCallReceipt:
    transport_kind: InventoryTransportKind
    model_id: str
    panel_png_digest: str
    panel_png_byte_count: int
    observer_contract_digest: str
    prompt_digest: str
    output_schema_digest: str
    request_digest: str
    response_digest: str
    transport_receipt_digest: str

    def __post_init__(self) -> None:
        if type(self.transport_kind) is not InventoryTransportKind:
            raise TypeError("inventory receipt transport kind differs")
        _code(self.model_id, "inventory receipt model ID")
        _digest(self.panel_png_digest, "inventory receipt panel digest")
        if type(self.panel_png_byte_count) is not int or self.panel_png_byte_count <= 0:
            raise PanelOwnerInventoryError("inventory receipt byte count differs")
        for label, value in (
            ("observer contract digest", self.observer_contract_digest),
            ("prompt digest", self.prompt_digest),
            ("output schema digest", self.output_schema_digest),
            ("request digest", self.request_digest),
            ("response digest", self.response_digest),
            ("transport receipt digest", self.transport_receipt_digest),
        ):
            _digest(value, label)
        request = panel_owner_inventory_request_data(
            transport_kind=self.transport_kind,
            model_id=self.model_id,
            panel_png_digest=self.panel_png_digest,
            panel_png_byte_count=self.panel_png_byte_count,
            observer_contract_digest=self.observer_contract_digest,
        )
        if self.prompt_digest != request["prompt_digest"]:
            raise PanelOwnerInventoryError("inventory receipt prompt differs")
        if self.output_schema_digest != request["output_schema_digest"]:
            raise PanelOwnerInventoryError("inventory receipt schema differs")
        if self.request_digest != canonical_digest(request):
            raise PanelOwnerInventoryError("inventory receipt request differs")

    @property
    def receipt_digest(self) -> str:
        return canonical_digest(self.to_data())

    def to_data(self) -> dict[str, object]:
        return {
            "schema": PANEL_OWNER_RECEIPT_SCHEMA,
            "transport_kind": self.transport_kind.value,
            "model_id": self.model_id,
            "panel_png_digest": self.panel_png_digest,
            "panel_png_byte_count": self.panel_png_byte_count,
            "observer_contract_digest": self.observer_contract_digest,
            "prompt_digest": self.prompt_digest,
            "output_schema_digest": self.output_schema_digest,
            "request_digest": self.request_digest,
            "response_digest": self.response_digest,
            "transport_receipt_digest": self.transport_receipt_digest,
        }

    @classmethod
    def from_data(cls, value: object) -> "PanelOwnerInventoryCallReceipt":
        raw = _fields(
            value,
            {
                "schema",
                "transport_kind",
                "model_id",
                "panel_png_digest",
                "panel_png_byte_count",
                "observer_contract_digest",
                "prompt_digest",
                "output_schema_digest",
                "request_digest",
                "response_digest",
                "transport_receipt_digest",
            },
            "panel owner receipt",
        )
        if raw["schema"] != PANEL_OWNER_RECEIPT_SCHEMA:
            raise PanelOwnerInventoryError("panel owner receipt schema differs")
        try:
            result = cls(
                InventoryTransportKind(raw["transport_kind"]),
                raw["model_id"],
                raw["panel_png_digest"],
                raw["panel_png_byte_count"],
                raw["observer_contract_digest"],
                raw["prompt_digest"],
                raw["output_schema_digest"],
                raw["request_digest"],
                raw["response_digest"],
                raw["transport_receipt_digest"],
            )
        except (TypeError, ValueError) as exc:
            if isinstance(exc, PanelOwnerInventoryError):
                raise
            raise PanelOwnerInventoryError("panel owner receipt value differs") from exc
        _canonical_roundtrip(result, raw, "panel owner receipt")
        return result


def bind_panel_owner_inventory_receipt(
    *,
    panel_png: bytes,
    observer_contract_digest: str,
    payload: Mapping[str, Any],
    transport_kind: InventoryTransportKind,
    model_id: str,
    transport_receipt_digest: str,
) -> PanelOwnerInventoryCallReceipt:
    _, panel_digest, byte_count = _panel_png_identity(panel_png)
    request = panel_owner_inventory_request_data(
        transport_kind=transport_kind,
        model_id=model_id,
        panel_png_digest=panel_digest,
        panel_png_byte_count=byte_count,
        observer_contract_digest=observer_contract_digest,
    )
    return PanelOwnerInventoryCallReceipt(
        transport_kind,
        model_id,
        panel_digest,
        byte_count,
        observer_contract_digest,
        request["prompt_digest"],
        request["output_schema_digest"],
        canonical_digest(request),
        canonical_digest(payload),
        transport_receipt_digest,
    )


@dataclass(frozen=True, slots=True)
class PanelOwnerInventoryArtifact:
    panel_png_digest: str
    panel_png_byte_count: int
    observer_contract_digest: str
    receipt: PanelOwnerInventoryCallReceipt
    response: ParsedPanelOwnerResponse
    status: InventoryStatus
    owners: tuple[PanelLocalOwner, ...]
    gap: InventoryGap | None

    def __post_init__(self) -> None:
        _digest(self.panel_png_digest, "artifact panel digest")
        if type(self.panel_png_byte_count) is not int or self.panel_png_byte_count <= 0:
            raise PanelOwnerInventoryError("artifact panel byte count differs")
        _digest(self.observer_contract_digest, "artifact observer contract digest")
        if type(self.receipt) is not PanelOwnerInventoryCallReceipt:
            raise TypeError("artifact receipt has the wrong type")
        if type(self.response) is not ParsedPanelOwnerResponse:
            raise TypeError("artifact response has the wrong type")
        if (
            self.receipt.panel_png_digest != self.panel_png_digest
            or self.receipt.panel_png_byte_count != self.panel_png_byte_count
            or self.receipt.observer_contract_digest != self.observer_contract_digest
        ):
            raise PanelOwnerInventoryError("artifact and receipt custody differ")
        if canonical_digest(self.response.to_data()) != self.receipt.response_digest:
            raise PanelOwnerInventoryError("artifact response and receipt differ")
        if type(self.status) is not InventoryStatus:
            raise TypeError("artifact status must be InventoryStatus")
        if type(self.owners) is not tuple or any(
            type(item) is not PanelLocalOwner for item in self.owners
        ):
            raise TypeError("artifact owners must be a PanelLocalOwner tuple")
        names = tuple(item.owner_id.value for item in self.owners)
        if names != tuple(sorted(names)) or len(names) != len(set(names)):
            raise PanelOwnerInventoryError("artifact owners must be unique and sorted")
        if self.status is InventoryStatus.COMPLETE:
            if self.gap is not None:
                raise PanelOwnerInventoryError("complete artifact cannot carry a gap")
        elif self.owners or type(self.gap) is not InventoryGap:
            raise PanelOwnerInventoryError("noncomplete artifact needs one gap")
        expected_status = None if self.gap is None else {
            InventoryGapKind.CAPACITY_EXCEEDED: InventoryStatus.CAPACITY_EXCEEDED,
            InventoryGapKind.TRANSPORT_ERROR: InventoryStatus.ERROR,
            InventoryGapKind.UNRESOLVED: InventoryStatus.UNRESOLVED,
            InventoryGapKind.SEMANTIC_COLLISION: InventoryStatus.UNRESOLVED,
            InventoryGapKind.PARENT_CYCLE: InventoryStatus.UNRESOLVED,
            InventoryGapKind.INVALID_PARENT: InventoryStatus.UNRESOLVED,
        }[self.gap.kind]
        if expected_status is not None and self.status is not expected_status:
            raise PanelOwnerInventoryError("artifact status and gap differ")
        replayed = canonicalize_panel_owner_response(self.response)
        if (
            replayed.status is not self.status
            or replayed.owners != self.owners
            or replayed.gap != self.gap
        ):
            raise PanelOwnerInventoryError("artifact canonical replay differs")

    @property
    def semantic_inventory_digest(self) -> str:
        return canonical_digest(self.semantic_data())

    @property
    def artifact_digest(self) -> str:
        return canonical_digest(self.to_data())

    def semantic_data(self) -> dict[str, object]:
        return {
            "schema": PANEL_OWNER_SEMANTIC_SCHEMA,
            "owner_semantics_digest": panel_owner_semantics_digest(),
            "inventory_contract_digest": panel_owner_inventory_contract_digest(),
            "panel_png_digest": self.panel_png_digest,
            "observer_contract_digest": self.observer_contract_digest,
            "status": self.status.value,
            "owners": [item.to_data() for item in self.owners],
            "gap": None if self.gap is None else self.gap.to_data(),
        }

    def to_data(self) -> dict[str, object]:
        return {
            "schema": PANEL_OWNER_ARTIFACT_SCHEMA,
            "owner_semantics_digest": panel_owner_semantics_digest(),
            "inventory_contract_digest": panel_owner_inventory_contract_digest(),
            "panel_png_digest": self.panel_png_digest,
            "panel_png_byte_count": self.panel_png_byte_count,
            "observer_contract_digest": self.observer_contract_digest,
            "prompt_digest": canonical_digest(panel_owner_inventory_prompt()),
            "output_schema_digest": canonical_digest(
                panel_owner_inventory_output_schema()
            ),
            "receipt": self.receipt.to_data(),
            "response": self.response.to_data(),
            "status": self.status.value,
            "owners": [item.to_data() for item in self.owners],
            "gap": None if self.gap is None else self.gap.to_data(),
            "semantic_inventory_digest": self.semantic_inventory_digest,
        }

    @classmethod
    def from_data(cls, value: object) -> "PanelOwnerInventoryArtifact":
        raw = _fields(
            value,
            {
                "schema",
                "owner_semantics_digest",
                "inventory_contract_digest",
                "panel_png_digest",
                "panel_png_byte_count",
                "observer_contract_digest",
                "prompt_digest",
                "output_schema_digest",
                "receipt",
                "response",
                "status",
                "owners",
                "gap",
                "semantic_inventory_digest",
            },
            "panel owner artifact",
        )
        if raw["schema"] != PANEL_OWNER_ARTIFACT_SCHEMA:
            raise PanelOwnerInventoryError("panel owner artifact schema differs")
        if raw["owner_semantics_digest"] != panel_owner_semantics_digest():
            raise PanelOwnerInventoryError("panel owner semantics differ")
        if raw["inventory_contract_digest"] != panel_owner_inventory_contract_digest():
            raise PanelOwnerInventoryError("panel owner inventory contract differs")
        if raw["prompt_digest"] != canonical_digest(panel_owner_inventory_prompt()):
            raise PanelOwnerInventoryError("panel owner prompt digest differs")
        if raw["output_schema_digest"] != canonical_digest(
            panel_owner_inventory_output_schema()
        ):
            raise PanelOwnerInventoryError("panel owner output schema digest differs")
        if type(raw["owners"]) is not list:
            raise PanelOwnerInventoryError("panel owner artifact owners must be a list")
        try:
            result = cls(
                raw["panel_png_digest"],
                raw["panel_png_byte_count"],
                raw["observer_contract_digest"],
                PanelOwnerInventoryCallReceipt.from_data(raw["receipt"]),
                parse_panel_owner_inventory_payload(raw["response"]),
                InventoryStatus(raw["status"]),
                tuple(PanelLocalOwner.from_data(item) for item in raw["owners"]),
                None if raw["gap"] is None else InventoryGap.from_data(raw["gap"]),
            )
        except (TypeError, ValueError) as exc:
            if isinstance(exc, PanelOwnerInventoryError):
                raise
            raise PanelOwnerInventoryError("panel owner artifact value differs") from exc
        if result.semantic_inventory_digest != raw["semantic_inventory_digest"]:
            raise PanelOwnerInventoryError("semantic inventory digest differs")
        _canonical_roundtrip(result, raw, "panel owner artifact")
        return result

    def to_owner_inventory(self) -> OwnerInventory:
        """Adapt a complete artifact to the generic ontology inventory type."""

        if self.status is not InventoryStatus.COMPLETE:
            raise PanelOwnerInventoryError("only a complete artifact has an owner inventory")
        protocol_binding = canonical_digest(
            {
                "inventory_contract_digest": panel_owner_inventory_contract_digest(),
                "observer_contract_digest": self.observer_contract_digest,
            }
        )
        return OwnerInventory(
            self.panel_png_digest,
            protocol_binding,
            EnumerationResolution.GRID16_FULL_PANEL,
            self.receipt.transport_receipt_digest,
            True,
            self.owners,
        )


def build_panel_owner_inventory_artifact(
    *,
    panel_png: bytes,
    observer_contract_digest: str,
    payload: Mapping[str, Any],
    receipt: PanelOwnerInventoryCallReceipt,
) -> PanelOwnerInventoryArtifact:
    """Validate exact custody, parse, and canonicalize one receipted response."""

    _, panel_digest, byte_count = _panel_png_identity(panel_png)
    _digest(observer_contract_digest, "observer contract digest")
    if type(receipt) is not PanelOwnerInventoryCallReceipt:
        raise TypeError("panel owner build receipt has the wrong type")
    if (
        receipt.panel_png_digest != panel_digest
        or receipt.panel_png_byte_count != byte_count
        or receipt.observer_contract_digest != observer_contract_digest
        or receipt.response_digest != canonical_digest(payload)
    ):
        raise PanelOwnerInventoryError("panel owner response custody differs")
    parsed = parse_panel_owner_inventory_payload(payload)
    canonical = canonicalize_panel_owner_response(parsed)
    return PanelOwnerInventoryArtifact(
        panel_digest,
        byte_count,
        observer_contract_digest,
        receipt,
        parsed,
        canonical.status,
        canonical.owners,
        canonical.gap,
    )


@dataclass(frozen=True, slots=True)
class PanelOwnerInventoryTransportResult:
    payload: Mapping[str, Any]
    receipt: PanelOwnerInventoryCallReceipt

    def __post_init__(self) -> None:
        if not isinstance(self.payload, Mapping):
            raise TypeError("panel owner transport payload must be a mapping")
        if type(self.receipt) is not PanelOwnerInventoryCallReceipt:
            raise TypeError("panel owner transport receipt has the wrong type")


class PanelOwnerInventoryTransport(Protocol):
    def __call__(
        self,
        *,
        prompt: str,
        panel_png: bytes,
        image_name: str,
        output_schema: Mapping[str, Any],
    ) -> PanelOwnerInventoryTransportResult: ...


def observe_panel_owner_inventory(
    *,
    panel_png: bytes,
    observer_contract_digest: str,
    transport: PanelOwnerInventoryTransport,
) -> PanelOwnerInventoryArtifact:
    """Execute one injectable, neutral, fixed-view receipted observation."""

    frozen_png, _, _ = _panel_png_identity(panel_png)
    _digest(observer_contract_digest, "observer contract digest")
    if not callable(transport):
        raise TypeError("panel owner transport must be callable")
    result = transport(
        prompt=panel_owner_inventory_prompt(),
        panel_png=frozen_png,
        image_name=PANEL_OWNER_NEUTRAL_IMAGE_NAME,
        output_schema=panel_owner_inventory_output_schema(),
    )
    if type(result) is not PanelOwnerInventoryTransportResult:
        raise TypeError("panel owner transport returned the wrong type")
    return build_panel_owner_inventory_artifact(
        panel_png=frozen_png,
        observer_contract_digest=observer_contract_digest,
        payload=result.payload,
        receipt=result.receipt,
    )

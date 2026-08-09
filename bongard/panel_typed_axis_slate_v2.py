"""Deterministic typed-axis support search with replayable evidence witnesses.

This module is deliberately below every model, transport, query, and ranking
layer.  It accepts twelve already-frozen support rows over eight fixed axes,
one support-only nomination (or a typed gap) per axis, and enumerates equality
singletons and pairs in Python.  A calibrated confidence set may induce a
runtime disposition, but its witnesses explicitly deny deterministic pixel
truth.  No prose predicate or Lean term is executable here.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re
from typing import Any, Mapping, Sequence, TypeAlias

from bongard.canonical import canonical_digest
from bongard.evidence import Disposition


AXIS_COUNT = 8
PRIMARY_ROW_COUNT = 6
CONTRAST_ROW_COUNT = 6
SUPPORT_ROW_COUNT = PRIMARY_ROW_COUNT + CONTRAST_ROW_COUNT
MAX_FORMULA_COUNT = AXIS_COUNT + AXIS_COUNT * (AXIS_COUNT - 1) // 2

CELL_SCHEMA = "gkm.bongard-typed-axis-cell.v2"
ROW_SCHEMA = "gkm.bongard-typed-axis-row.v2"
MATRIX_SCHEMA = "gkm.bongard-typed-axis-support-matrix.v2"
NOMINATION_SCHEMA = "gkm.bongard-typed-axis-nomination.v2"
NOMINATION_SLATE_SCHEMA = "gkm.bongard-typed-axis-nomination-slate.v2"
ATOM_SCHEMA = "gkm.bongard-typed-axis-equality-atom.v2"
WITNESS_SCHEMA = "gkm.bongard-typed-axis-evidence-witness.v2"
ROW_EVALUATION_SCHEMA = "gkm.bongard-typed-axis-formula-row-evaluation.v2"
FORMULA_SCHEMA = "gkm.bongard-typed-axis-formula-evaluation.v2"
EMPTY_GAP_SCHEMA = "gkm.bongard-typed-axis-empty-gap.v2"
INVENTORY_SCHEMA = "gkm.bongard-typed-axis-inventory.v2"
ALGORITHM_ID = "bongard.typed-axis/equality-singletons-pairs-support-only-v2"

_ADDRESS = re.compile(r"sha256:[0-9a-f]{64}\Z")
_KEY = re.compile(r"[A-Za-z0-9][A-Za-z0-9_./:-]{0,255}\Z")
_CODE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")


class TypedAxisSlateError(ValueError):
    """A typed support value or deterministic replay differs."""


class Axis(str, Enum):
    TOPOLOGY = "topology"
    COMPONENT_COUNT = "component_count"
    STRAIGHT_ACTION_COUNT = "straight_action_count"
    PRIMITIVE_MIX_OR_ARC_COUNT = "primitive_mix_or_arc_count"
    TURNING_CONVEXITY = "turning_convexity"
    SYMMETRY = "symmetry"
    ASPECT_ORIENTATION = "aspect_orientation"
    TEXTURE = "texture"


AXES = tuple(Axis)


class EvidenceKind(str, Enum):
    PYTHON_EXACT = "python_exact"
    CALIBRATED_SET = "calibrated_set"
    GAP = "gap"
    ERROR = "error"


class SupportSide(str, Enum):
    PRIMARY = "primary"
    CONTRAST = "contrast"


AxisValue: TypeAlias = int | str


# Each axis is a finite, versioned state space.  Counts use literal integers;
# categoricals use literal strings.  ``primitive_mix_or_arc_count`` combines
# primitive mix and bounded arc count into one mutually exclusive value.
AXIS_DOMAINS: Mapping[Axis, tuple[AxisValue, ...]] = {
    Axis.TOPOLOGY: ("open", "closed", "mixed_open_closed"),
    Axis.COMPONENT_COUNT: tuple(range(10)),
    Axis.STRAIGHT_ACTION_COUNT: tuple(range(10)),
    Axis.PRIMITIVE_MIX_OR_ARC_COUNT: (
        "no_carrier_actions",
        "straight_only",
        *(f"arc_only_{count}" for count in range(1, 10)),
        *(f"mixed_{count}_arcs" for count in range(1, 10)),
    ),
    Axis.TURNING_CONVEXITY: (
        "convex_turning",
        "concave_turning",
        "mixed_turning",
        "not_applicable",
    ),
    Axis.SYMMETRY: ("none", "reflection", "rotation", "reflection_and_rotation"),
    Axis.ASPECT_ORIENTATION: (
        "compact",
        "elongated_horizontal",
        "elongated_vertical",
        "elongated_oblique_positive",
        "elongated_oblique_negative",
    ),
    Axis.TEXTURE: ("plain", "dotted", "dashed", "mixed_texture"),
}


def _fields(value: object, expected: set[str], label: str) -> Mapping[str, Any]:
    if (
        not isinstance(value, Mapping)
        or any(type(key) is not str for key in value)
        or set(value) != expected
    ):
        raise TypedAxisSlateError(f"{label} fields differ")
    return value


def _address(value: object, label: str) -> str:
    if type(value) is not str or _ADDRESS.fullmatch(value) is None:
        raise TypedAxisSlateError(f"{label} must be a sha256: address")
    return value


def _key(value: object, label: str) -> str:
    if type(value) is not str or _KEY.fullmatch(value) is None:
        raise TypedAxisSlateError(f"{label} must be a bounded key")
    return value


def _code(value: object, label: str) -> str:
    if type(value) is not str or _CODE.fullmatch(value) is None:
        raise TypedAxisSlateError(f"{label} must be a bounded code")
    return value


def _axis(value: object) -> Axis:
    try:
        return Axis(value)
    except (TypeError, ValueError) as exc:
        raise TypedAxisSlateError("axis differs") from exc


def _disposition(value: object) -> Disposition:
    try:
        return Disposition(value)
    except (TypeError, ValueError) as exc:
        raise TypedAxisSlateError("disposition differs") from exc


def _same_value(left: object, right: object) -> bool:
    return type(left) is type(right) and left == right


def _domain_index(axis: Axis, value: object) -> int:
    for index, candidate in enumerate(AXIS_DOMAINS[axis]):
        if _same_value(candidate, value):
            return index
    raise TypedAxisSlateError(f"value lies outside closed {axis.value} domain")


def _value(axis: Axis, value: object) -> AxisValue:
    return AXIS_DOMAINS[axis][_domain_index(axis, value)]


def _ordered_values(axis: Axis, values: Sequence[object]) -> tuple[AxisValue, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise TypedAxisSlateError("possible values must be a sequence")
    checked = tuple(_value(axis, item) for item in values)
    if len(checked) != len({_domain_index(axis, item) for item in checked}):
        raise TypedAxisSlateError("possible values must be unique")
    if tuple(sorted(checked, key=lambda item: _domain_index(axis, item))) != checked:
        raise TypedAxisSlateError("possible values must follow closed-domain order")
    return checked


@dataclass(frozen=True, slots=True)
class TypedAxisCell:
    axis: Axis
    evidence_kind: EvidenceKind
    possible_values: tuple[AxisValue, ...]
    observer_protocol_digest: str
    calibration_grant_address: str | None = None
    gap_reason_code: str | None = None
    error_code: str | None = None

    def __post_init__(self) -> None:
        if type(self.axis) is not Axis or type(self.evidence_kind) is not EvidenceKind:
            raise TypeError("typed cell needs exact enums")
        if type(self.possible_values) is not tuple:
            raise TypeError("typed cell possible values need tuple")
        values = _ordered_values(self.axis, self.possible_values)
        _address(self.observer_protocol_digest, "observer protocol digest")
        if self.evidence_kind is EvidenceKind.PYTHON_EXACT:
            if len(values) != 1 or any(
                item is not None
                for item in (
                    self.calibration_grant_address,
                    self.gap_reason_code,
                    self.error_code,
                )
            ):
                raise TypedAxisSlateError("Python-exact cell fields differ")
        elif self.evidence_kind is EvidenceKind.CALIBRATED_SET:
            if (
                not values
                or self.calibration_grant_address is None
                or self.gap_reason_code is not None
                or self.error_code is not None
            ):
                raise TypedAxisSlateError("calibrated-set cell fields differ")
            _address(self.calibration_grant_address, "calibration grant address")
        elif self.evidence_kind is EvidenceKind.GAP:
            if (
                values
                or self.calibration_grant_address is not None
                or self.gap_reason_code is None
                or self.error_code is not None
            ):
                raise TypedAxisSlateError("gap cell fields differ")
            _code(self.gap_reason_code, "gap reason code")
        else:
            if (
                values
                or self.calibration_grant_address is not None
                or self.gap_reason_code is not None
                or self.error_code is None
            ):
                raise TypedAxisSlateError("error cell fields differ")
            _code(self.error_code, "cell error code")

    @classmethod
    def python_exact(
        cls, axis: Axis, value: AxisValue, observer_protocol_digest: str
    ) -> "TypedAxisCell":
        return cls(axis, EvidenceKind.PYTHON_EXACT, (value,), observer_protocol_digest)

    @classmethod
    def calibrated_set(
        cls,
        axis: Axis,
        values: Sequence[AxisValue],
        observer_protocol_digest: str,
        calibration_grant_address: str,
    ) -> "TypedAxisCell":
        ordered = tuple(
            sorted(values, key=lambda item: _domain_index(axis, item))
        )
        return cls(
            axis,
            EvidenceKind.CALIBRATED_SET,
            ordered,
            observer_protocol_digest,
            calibration_grant_address,
        )

    @classmethod
    def gap(
        cls, axis: Axis, observer_protocol_digest: str, reason_code: str
    ) -> "TypedAxisCell":
        return cls(
            axis,
            EvidenceKind.GAP,
            (),
            observer_protocol_digest,
            gap_reason_code=reason_code,
        )

    @classmethod
    def error(
        cls, axis: Axis, observer_protocol_digest: str, error_code: str
    ) -> "TypedAxisCell":
        return cls(
            axis,
            EvidenceKind.ERROR,
            (),
            observer_protocol_digest,
            error_code=error_code,
        )

    def equality_disposition(self, nominated_value: AxisValue) -> Disposition:
        value = _value(self.axis, nominated_value)
        if self.evidence_kind is EvidenceKind.ERROR:
            return Disposition.ERROR
        if self.evidence_kind is EvidenceKind.GAP:
            return Disposition.INDETERMINATE
        included = any(_same_value(item, value) for item in self.possible_values)
        if not included:
            return Disposition.CERTIFIED_ABSENT
        if len(self.possible_values) == 1:
            return Disposition.PRESENT
        return Disposition.INDETERMINATE

    def to_data(self) -> dict[str, object]:
        return {
            "schema": CELL_SCHEMA,
            "axis": self.axis.value,
            "evidence_kind": self.evidence_kind.value,
            "possible_values": list(self.possible_values),
            "observer_protocol_digest": self.observer_protocol_digest,
            "calibration_grant_address": self.calibration_grant_address,
            "gap_reason_code": self.gap_reason_code,
            "error_code": self.error_code,
        }

    @classmethod
    def from_data(cls, value: object) -> "TypedAxisCell":
        raw = _fields(
            value,
            {
                "schema",
                "axis",
                "evidence_kind",
                "possible_values",
                "observer_protocol_digest",
                "calibration_grant_address",
                "gap_reason_code",
                "error_code",
            },
            "typed axis cell",
        )
        if raw["schema"] != CELL_SCHEMA or type(raw["possible_values"]) is not list:
            raise TypedAxisSlateError("typed axis cell schema differs")
        try:
            kind = EvidenceKind(raw["evidence_kind"])
        except (TypeError, ValueError) as exc:
            raise TypedAxisSlateError("evidence kind differs") from exc
        result = cls(
            _axis(raw["axis"]),
            kind,
            tuple(raw["possible_values"]),
            raw["observer_protocol_digest"],
            raw["calibration_grant_address"],
            raw["gap_reason_code"],
            raw["error_code"],
        )
        if result.to_data() != dict(raw):
            raise TypedAxisSlateError("typed axis cell is not canonical")
        return result


@dataclass(frozen=True, slots=True)
class TypedSupportRow:
    row_key: str
    side: SupportSide
    cells: tuple[TypedAxisCell, ...]

    def __post_init__(self) -> None:
        _key(self.row_key, "support row key")
        if type(self.side) is not SupportSide or type(self.cells) is not tuple:
            raise TypeError("support row needs exact side and tuple cells")
        if (
            len(self.cells) != AXIS_COUNT
            or any(type(item) is not TypedAxisCell for item in self.cells)
            or tuple(item.axis for item in self.cells) != AXES
        ):
            raise TypedAxisSlateError("support row must contain the fixed eight axes")

    def cell(self, axis: Axis) -> TypedAxisCell:
        if type(axis) is not Axis:
            raise TypeError("cell lookup needs exact Axis")
        return self.cells[AXES.index(axis)]

    def to_data(self) -> dict[str, object]:
        return {
            "schema": ROW_SCHEMA,
            "row_key": self.row_key,
            "side": self.side.value,
            "cells": [item.to_data() for item in self.cells],
        }

    @classmethod
    def from_data(cls, value: object) -> "TypedSupportRow":
        raw = _fields(value, {"schema", "row_key", "side", "cells"}, "support row")
        if raw["schema"] != ROW_SCHEMA or type(raw["cells"]) is not list:
            raise TypedAxisSlateError("support row schema differs")
        try:
            side = SupportSide(raw["side"])
        except (TypeError, ValueError) as exc:
            raise TypedAxisSlateError("support side differs") from exc
        result = cls(
            raw["row_key"],
            side,
            tuple(TypedAxisCell.from_data(item) for item in raw["cells"]),
        )
        if result.to_data() != dict(raw):
            raise TypedAxisSlateError("support row is not canonical")
        return result


@dataclass(frozen=True, slots=True)
class TypedSupportMatrix:
    rows: tuple[TypedSupportRow, ...]
    observer_protocol_digest: str

    def __post_init__(self) -> None:
        if type(self.rows) is not tuple or len(self.rows) != SUPPORT_ROW_COUNT:
            raise TypedAxisSlateError("matrix needs exactly twelve ordered rows")
        if any(type(item) is not TypedSupportRow for item in self.rows):
            raise TypeError("matrix rows must be exact TypedSupportRow")
        if tuple(row.side for row in self.rows) != (
            (SupportSide.PRIMARY,) * PRIMARY_ROW_COUNT
            + (SupportSide.CONTRAST,) * CONTRAST_ROW_COUNT
        ):
            raise TypedAxisSlateError("matrix row sides must be six primary then six contrast")
        if len({row.row_key for row in self.rows}) != SUPPORT_ROW_COUNT:
            raise TypedAxisSlateError("matrix row keys must be unique")
        _address(self.observer_protocol_digest, "matrix observer protocol digest")
        if any(
            cell.observer_protocol_digest != self.observer_protocol_digest
            for row in self.rows
            for cell in row.cells
        ):
            raise TypedAxisSlateError("matrix mixes observer protocols")

    @classmethod
    def freeze(cls, rows: Sequence[TypedSupportRow]) -> "TypedSupportMatrix":
        frozen = tuple(rows)
        if not frozen:
            raise TypedAxisSlateError("matrix rows are empty")
        return cls(frozen, frozen[0].cells[0].observer_protocol_digest)

    @property
    def matrix_address(self) -> str:
        return "sha256:" + canonical_digest(self.to_data())

    def to_data(self) -> dict[str, object]:
        return {
            "schema": MATRIX_SCHEMA,
            "observer_protocol_digest": self.observer_protocol_digest,
            "axis_order": [axis.value for axis in AXES],
            "closed_domains": {
                axis.value: list(AXIS_DOMAINS[axis]) for axis in AXES
            },
            "rows": [row.to_data() for row in self.rows],
            "row_count": SUPPORT_ROW_COUNT,
            "primary_row_count": PRIMARY_ROW_COUNT,
            "contrast_row_count": CONTRAST_ROW_COUNT,
        }

    @classmethod
    def from_data(cls, value: object) -> "TypedSupportMatrix":
        raw = _fields(
            value,
            {
                "schema",
                "observer_protocol_digest",
                "axis_order",
                "closed_domains",
                "rows",
                "row_count",
                "primary_row_count",
                "contrast_row_count",
            },
            "typed support matrix",
        )
        if (
            raw["schema"] != MATRIX_SCHEMA
            or raw["axis_order"] != [axis.value for axis in AXES]
            or raw["closed_domains"]
            != {axis.value: list(AXIS_DOMAINS[axis]) for axis in AXES}
            or type(raw["rows"]) is not list
            or raw["row_count"] != SUPPORT_ROW_COUNT
            or raw["primary_row_count"] != PRIMARY_ROW_COUNT
            or raw["contrast_row_count"] != CONTRAST_ROW_COUNT
        ):
            raise TypedAxisSlateError("typed support matrix policy differs")
        result = cls(
            tuple(TypedSupportRow.from_data(item) for item in raw["rows"]),
            raw["observer_protocol_digest"],
        )
        if result.to_data() != dict(raw):
            raise TypedAxisSlateError("typed support matrix is not canonical")
        return result


@dataclass(frozen=True, slots=True)
class AxisNomination:
    axis: Axis
    value: AxisValue | None
    gap_reason_code: str | None = None

    def __post_init__(self) -> None:
        if type(self.axis) is not Axis:
            raise TypeError("nomination needs exact Axis")
        if self.value is None:
            if self.gap_reason_code is None:
                raise TypedAxisSlateError("nomination gap needs a reason code")
            _code(self.gap_reason_code, "nomination gap reason")
        else:
            _value(self.axis, self.value)
            if self.gap_reason_code is not None:
                raise TypedAxisSlateError("available nomination cannot name a gap")

    @classmethod
    def nominate(cls, axis: Axis, value: AxisValue) -> "AxisNomination":
        return cls(axis, value)

    @classmethod
    def gap(cls, axis: Axis, reason_code: str) -> "AxisNomination":
        return cls(axis, None, reason_code)

    def to_data(self) -> dict[str, object]:
        return {
            "schema": NOMINATION_SCHEMA,
            "axis": self.axis.value,
            "status": "nominated" if self.value is not None else "gap",
            "value": self.value,
            "gap_reason_code": self.gap_reason_code,
        }

    @classmethod
    def from_data(cls, value: object) -> "AxisNomination":
        raw = _fields(
            value,
            {"schema", "axis", "status", "value", "gap_reason_code"},
            "axis nomination",
        )
        if raw["schema"] != NOMINATION_SCHEMA or raw["status"] not in {
            "nominated",
            "gap",
        }:
            raise TypedAxisSlateError("axis nomination schema differs")
        result = cls(_axis(raw["axis"]), raw["value"], raw["gap_reason_code"])
        if result.to_data() != dict(raw):
            raise TypedAxisSlateError("axis nomination is not canonical")
        return result


@dataclass(frozen=True, slots=True)
class TypedNominationSlate:
    support_matrix_address: str
    nominations: tuple[AxisNomination, ...]

    def __post_init__(self) -> None:
        _address(self.support_matrix_address, "nomination support matrix address")
        if (
            type(self.nominations) is not tuple
            or len(self.nominations) != AXIS_COUNT
            or any(type(item) is not AxisNomination for item in self.nominations)
            or tuple(item.axis for item in self.nominations) != AXES
        ):
            raise TypedAxisSlateError(
                "slate needs one nomination value or typed gap per fixed axis"
            )

    @classmethod
    def freeze(
        cls, matrix: TypedSupportMatrix, nominations: Sequence[AxisNomination]
    ) -> "TypedNominationSlate":
        if type(matrix) is not TypedSupportMatrix:
            raise TypeError("nomination slate needs exact matrix")
        return cls(matrix.matrix_address, tuple(nominations))

    @property
    def available(self) -> tuple[AxisNomination, ...]:
        return tuple(item for item in self.nominations if item.value is not None)

    def to_data(self) -> dict[str, object]:
        return {
            "schema": NOMINATION_SLATE_SCHEMA,
            "support_matrix_address": self.support_matrix_address,
            "axis_order": [axis.value for axis in AXES],
            "nominations": [item.to_data() for item in self.nominations],
            "support_only": True,
            "query_rows_seen": 0,
            "polarity_choice_present": False,
        }

    @classmethod
    def from_data(cls, value: object) -> "TypedNominationSlate":
        raw = _fields(
            value,
            {
                "schema",
                "support_matrix_address",
                "axis_order",
                "nominations",
                "support_only",
                "query_rows_seen",
                "polarity_choice_present",
            },
            "typed nomination slate",
        )
        if (
            raw["schema"] != NOMINATION_SLATE_SCHEMA
            or raw["axis_order"] != [axis.value for axis in AXES]
            or type(raw["nominations"]) is not list
            or raw["support_only"] is not True
            or raw["query_rows_seen"] != 0
            or raw["polarity_choice_present"] is not False
        ):
            raise TypedAxisSlateError("typed nomination slate policy differs")
        result = cls(
            raw["support_matrix_address"],
            tuple(AxisNomination.from_data(item) for item in raw["nominations"]),
        )
        if result.to_data() != dict(raw):
            raise TypedAxisSlateError("typed nomination slate is not canonical")
        return result


@dataclass(frozen=True, slots=True)
class EqualityAtom:
    axis: Axis
    value: AxisValue

    def __post_init__(self) -> None:
        if type(self.axis) is not Axis:
            raise TypeError("equality atom needs exact Axis")
        _value(self.axis, self.value)

    def to_data(self) -> dict[str, object]:
        return {"schema": ATOM_SCHEMA, "axis": self.axis.value, "value": self.value}

    @classmethod
    def from_data(cls, value: object) -> "EqualityAtom":
        raw = _fields(value, {"schema", "axis", "value"}, "equality atom")
        if raw["schema"] != ATOM_SCHEMA:
            raise TypedAxisSlateError("equality atom schema differs")
        result = cls(_axis(raw["axis"]), raw["value"])
        if result.to_data() != dict(raw):
            raise TypedAxisSlateError("equality atom is not canonical")
        return result


@dataclass(frozen=True, slots=True)
class EvidenceWitness:
    axis: Axis
    nominated_value: AxisValue
    disposition: Disposition
    evidence_kind: EvidenceKind
    possible_values: tuple[AxisValue, ...]
    observer_protocol_digest: str
    calibration_grant_address: str | None
    deterministic_pixel_truth_claimed: bool
    basis_code: str

    @classmethod
    def evaluate(cls, cell: TypedAxisCell, value: AxisValue) -> "EvidenceWitness":
        disposition = cell.equality_disposition(value)
        if cell.evidence_kind is EvidenceKind.PYTHON_EXACT:
            basis = (
                "python_exact_match"
                if disposition is Disposition.PRESENT
                else "python_exact_exclusion"
            )
            deterministic = True
        elif cell.evidence_kind is EvidenceKind.CALIBRATED_SET:
            basis = {
                Disposition.PRESENT: "calibrated_singleton_match",
                Disposition.CERTIFIED_ABSENT: "calibrated_set_exclusion",
                Disposition.INDETERMINATE: "calibrated_set_contains_alternatives",
            }[disposition]
            deterministic = False
        elif cell.evidence_kind is EvidenceKind.GAP:
            basis = f"gap:{cell.gap_reason_code}"
            deterministic = False
        else:
            basis = f"error:{cell.error_code}"
            deterministic = False
        return cls(
            cell.axis,
            _value(cell.axis, value),
            disposition,
            cell.evidence_kind,
            cell.possible_values,
            cell.observer_protocol_digest,
            cell.calibration_grant_address,
            deterministic,
            basis,
        )

    def __post_init__(self) -> None:
        if (
            type(self.axis) is not Axis
            or type(self.disposition) is not Disposition
            or type(self.evidence_kind) is not EvidenceKind
            or type(self.possible_values) is not tuple
            or type(self.deterministic_pixel_truth_claimed) is not bool
        ):
            raise TypeError("evidence witness fields need exact types")
        _value(self.axis, self.nominated_value)
        _ordered_values(self.axis, self.possible_values)
        _address(self.observer_protocol_digest, "witness observer protocol digest")
        _code(self.basis_code.replace(":", "_"), "witness basis")
        if self.evidence_kind is EvidenceKind.CALIBRATED_SET:
            if self.calibration_grant_address is None:
                raise TypedAxisSlateError("calibrated witness lacks grant")
            _address(self.calibration_grant_address, "witness calibration grant")
            if self.deterministic_pixel_truth_claimed:
                raise TypedAxisSlateError("calibrated set cannot claim deterministic truth")
        elif self.calibration_grant_address is not None:
            raise TypedAxisSlateError("non-calibrated witness names a grant")
        if self.evidence_kind is not EvidenceKind.PYTHON_EXACT and self.deterministic_pixel_truth_claimed:
            raise TypedAxisSlateError("non-exact witness claims deterministic truth")

    def to_data(self) -> dict[str, object]:
        return {
            "schema": WITNESS_SCHEMA,
            "axis": self.axis.value,
            "nominated_value": self.nominated_value,
            "disposition": self.disposition.value,
            "evidence_kind": self.evidence_kind.value,
            "possible_values": list(self.possible_values),
            "observer_protocol_digest": self.observer_protocol_digest,
            "calibration_grant_address": self.calibration_grant_address,
            "deterministic_pixel_truth_claimed": self.deterministic_pixel_truth_claimed,
            "basis_code": self.basis_code,
        }


def _conjunction_disposition(witnesses: tuple[EvidenceWitness, ...]) -> Disposition:
    states = tuple(item.disposition for item in witnesses)
    if Disposition.ERROR in states:
        return Disposition.ERROR
    if Disposition.CERTIFIED_ABSENT in states:
        return Disposition.CERTIFIED_ABSENT
    if all(item is Disposition.PRESENT for item in states):
        return Disposition.PRESENT
    return Disposition.INDETERMINATE


@dataclass(frozen=True, slots=True)
class FormulaRowEvaluation:
    formula_id: str
    row_key: str
    side: SupportSide
    atom_witnesses: tuple[EvidenceWitness, ...]
    disposition: Disposition
    failure_witnesses: tuple[EvidenceWitness, ...]

    @classmethod
    def evaluate(
        cls, formula_id: str, atoms: tuple[EqualityAtom, ...], row: TypedSupportRow
    ) -> "FormulaRowEvaluation":
        witnesses = tuple(
            EvidenceWitness.evaluate(row.cell(atom.axis), atom.value) for atom in atoms
        )
        disposition = _conjunction_disposition(witnesses)
        failures = tuple(
            item for item in witnesses if item.disposition is disposition
        ) if disposition is not Disposition.PRESENT else ()
        return cls(formula_id, row.row_key, row.side, witnesses, disposition, failures)

    def __post_init__(self) -> None:
        _key(self.formula_id, "formula id")
        _key(self.row_key, "formula row key")
        if (
            type(self.side) is not SupportSide
            or type(self.atom_witnesses) is not tuple
            or not 1 <= len(self.atom_witnesses) <= 2
            or any(type(item) is not EvidenceWitness for item in self.atom_witnesses)
            or type(self.disposition) is not Disposition
            or type(self.failure_witnesses) is not tuple
        ):
            raise TypedAxisSlateError("formula row evaluation fields differ")
        expected = _conjunction_disposition(self.atom_witnesses)
        expected_failures = tuple(
            item for item in self.atom_witnesses if item.disposition is expected
        ) if expected is not Disposition.PRESENT else ()
        if self.disposition is not expected or self.failure_witnesses != expected_failures:
            raise TypedAxisSlateError("formula row disposition or failures differ")

    def to_data(self) -> dict[str, object]:
        return {
            "schema": ROW_EVALUATION_SCHEMA,
            "formula_id": self.formula_id,
            "row_key": self.row_key,
            "side": self.side.value,
            "atom_witnesses": [item.to_data() for item in self.atom_witnesses],
            "disposition": self.disposition.value,
            "failure_witnesses": [item.to_data() for item in self.failure_witnesses],
        }


@dataclass(frozen=True, slots=True)
class FormulaEvaluation:
    formula_id: str
    atoms: tuple[EqualityAtom, ...]
    rows: tuple[FormulaRowEvaluation, ...]
    primary_counts: tuple[int, int, int, int]
    contrast_counts: tuple[int, int, int, int]
    admitted: bool
    admission_failure_codes: tuple[str, ...]

    @staticmethod
    def _counts(rows: Sequence[FormulaRowEvaluation]) -> tuple[int, int, int, int]:
        return tuple(
            sum(item.disposition is state for item in rows)
            for state in (
                Disposition.PRESENT,
                Disposition.CERTIFIED_ABSENT,
                Disposition.INDETERMINATE,
                Disposition.ERROR,
            )
        )  # type: ignore[return-value]

    @classmethod
    def evaluate(
        cls,
        formula_id: str,
        atoms: tuple[EqualityAtom, ...],
        matrix: TypedSupportMatrix,
    ) -> "FormulaEvaluation":
        rows = tuple(
            FormulaRowEvaluation.evaluate(formula_id, atoms, row) for row in matrix.rows
        )
        primary = cls._counts(rows[:PRIMARY_ROW_COUNT])
        contrast = cls._counts(rows[PRIMARY_ROW_COUNT:])
        failures: list[str] = []
        if primary[0] < 5:
            failures.append("primary_present_below_5")
        if primary[1] != 0:
            failures.append("primary_absent_nonzero")
        if primary[2] > 1:
            failures.append("primary_indeterminate_above_1")
        if primary[3] != 0:
            failures.append("primary_error_nonzero")
        if contrast[1] < 5:
            failures.append("contrast_absent_below_5")
        if contrast[0] != 0:
            failures.append("contrast_present_nonzero")
        if contrast[2] > 1:
            failures.append("contrast_indeterminate_above_1")
        if contrast[3] != 0:
            failures.append("contrast_error_nonzero")
        return cls(
            formula_id,
            atoms,
            rows,
            primary,
            contrast,
            not failures,
            tuple(failures),
        )

    def __post_init__(self) -> None:
        _key(self.formula_id, "formula id")
        if (
            type(self.atoms) is not tuple
            or not 1 <= len(self.atoms) <= 2
            or any(type(item) is not EqualityAtom for item in self.atoms)
            or len({item.axis for item in self.atoms}) != len(self.atoms)
            or tuple(sorted(self.atoms, key=lambda item: AXES.index(item.axis))) != self.atoms
            or type(self.rows) is not tuple
            or len(self.rows) != SUPPORT_ROW_COUNT
            or any(type(item) is not FormulaRowEvaluation for item in self.rows)
            or any(item.formula_id != self.formula_id for item in self.rows)
            or type(self.primary_counts) is not tuple
            or type(self.contrast_counts) is not tuple
            or type(self.admitted) is not bool
            or type(self.admission_failure_codes) is not tuple
        ):
            raise TypedAxisSlateError("formula evaluation fields differ")
        expected_primary = self._counts(self.rows[:PRIMARY_ROW_COUNT])
        expected_contrast = self._counts(self.rows[PRIMARY_ROW_COUNT:])
        if self.primary_counts != expected_primary or self.contrast_counts != expected_contrast:
            raise TypedAxisSlateError("formula disposition counts differ")
        if self.admitted is not (not self.admission_failure_codes):
            raise TypedAxisSlateError("formula admission fields differ")
        for code in self.admission_failure_codes:
            _code(code, "admission failure code")

    def to_data(self) -> dict[str, object]:
        labels = ["present", "certified_absent", "indeterminate", "error"]
        return {
            "schema": FORMULA_SCHEMA,
            "formula_id": self.formula_id,
            "atoms": [item.to_data() for item in self.atoms],
            "rows": [item.to_data() for item in self.rows],
            "primary_counts": dict(zip(labels, self.primary_counts, strict=True)),
            "contrast_counts": dict(zip(labels, self.contrast_counts, strict=True)),
            "admitted": self.admitted,
            "admission_failure_codes": list(self.admission_failure_codes),
        }


@dataclass(frozen=True, slots=True)
class TypedEmptyGap:
    nomination_gap_axes: tuple[Axis, ...]
    evaluated_formula_count: int
    rejected_formula_ids: tuple[str, ...]
    reason_code: str = "no_formula_admitted"

    def __post_init__(self) -> None:
        if (
            type(self.nomination_gap_axes) is not tuple
            or tuple(sorted(self.nomination_gap_axes, key=AXES.index))
            != self.nomination_gap_axes
            or len(set(self.nomination_gap_axes)) != len(self.nomination_gap_axes)
            or any(type(item) is not Axis for item in self.nomination_gap_axes)
            or type(self.evaluated_formula_count) is not int
            or self.evaluated_formula_count < 0
            or type(self.rejected_formula_ids) is not tuple
            or len(self.rejected_formula_ids) != self.evaluated_formula_count
        ):
            raise TypedAxisSlateError("typed empty gap fields differ")
        _code(self.reason_code, "empty gap reason")

    def to_data(self) -> dict[str, object]:
        return {
            "schema": EMPTY_GAP_SCHEMA,
            "reason_code": self.reason_code,
            "nomination_gap_axes": [item.value for item in self.nomination_gap_axes],
            "evaluated_formula_count": self.evaluated_formula_count,
            "rejected_formula_ids": list(self.rejected_formula_ids),
        }


def _enumerate_atoms(slate: TypedNominationSlate) -> tuple[tuple[EqualityAtom, ...], ...]:
    atoms = tuple(EqualityAtom(item.axis, item.value) for item in slate.available)
    return tuple((item,) for item in atoms) + tuple(
        (atoms[left], atoms[right])
        for left in range(len(atoms))
        for right in range(left + 1, len(atoms))
    )


@dataclass(frozen=True, slots=True)
class TypedAxisInventory:
    matrix: TypedSupportMatrix
    nominations: TypedNominationSlate
    formulas: tuple[FormulaEvaluation, ...]
    admitted_formula_ids: tuple[str, ...]
    empty_gap: TypedEmptyGap | None

    @classmethod
    def derive(
        cls, matrix: TypedSupportMatrix, nominations: TypedNominationSlate
    ) -> "TypedAxisInventory":
        if type(matrix) is not TypedSupportMatrix or type(nominations) is not TypedNominationSlate:
            raise TypeError("inventory derivation needs exact inputs")
        if nominations.support_matrix_address != matrix.matrix_address:
            raise TypedAxisSlateError("nomination slate is bound to another support matrix")
        formula_atoms = _enumerate_atoms(nominations)
        formulas = tuple(
            FormulaEvaluation.evaluate(f"formula_{index:02d}", atoms, matrix)
            for index, atoms in enumerate(formula_atoms)
        )
        admitted = tuple(item.formula_id for item in formulas if item.admitted)
        gap = None
        if not admitted:
            gap = TypedEmptyGap(
                tuple(item.axis for item in nominations.nominations if item.value is None),
                len(formulas),
                tuple(item.formula_id for item in formulas),
            )
        return cls(matrix, nominations, formulas, admitted, gap)

    def __post_init__(self) -> None:
        if (
            type(self.matrix) is not TypedSupportMatrix
            or type(self.nominations) is not TypedNominationSlate
            or self.nominations.support_matrix_address != self.matrix.matrix_address
            or type(self.formulas) is not tuple
            or len(self.formulas) > MAX_FORMULA_COUNT
            or any(type(item) is not FormulaEvaluation for item in self.formulas)
            or tuple(item.formula_id for item in self.formulas)
            != tuple(f"formula_{index:02d}" for index in range(len(self.formulas)))
            or type(self.admitted_formula_ids) is not tuple
            or self.admitted_formula_ids
            != tuple(item.formula_id for item in self.formulas if item.admitted)
        ):
            raise TypedAxisSlateError("typed axis inventory fields differ")
        if self.admitted_formula_ids:
            if self.empty_gap is not None:
                raise TypedAxisSlateError("admitted inventory cannot carry empty gap")
        elif type(self.empty_gap) is not TypedEmptyGap:
            raise TypedAxisSlateError("empty inventory must carry typed gap")

    @property
    def inventory_address(self) -> str:
        return "sha256:" + canonical_digest(self.to_data())

    def to_data(self) -> dict[str, object]:
        return {
            "schema": INVENTORY_SCHEMA,
            "algorithm_id": ALGORITHM_ID,
            "matrix": self.matrix.to_data(),
            "nominations": self.nominations.to_data(),
            "formulas": [item.to_data() for item in self.formulas],
            "formula_count": len(self.formulas),
            "maximum_formula_count": MAX_FORMULA_COUNT,
            "admitted_formula_ids": list(self.admitted_formula_ids),
            "empty_gap": None if self.empty_gap is None else self.empty_gap.to_data(),
            "formula_language": "positive_equality_singleton_or_pair",
            "conjunction_precedence": [
                "error",
                "certified_absent",
                "all_present",
                "indeterminate",
            ],
            "query_rows_seen": 0,
            "model_calls_for_derivation_or_replay": 0,
            "ranking_present": False,
            "negation_present": False,
            "lean_present": False,
            "calibrated_sets_claim_deterministic_pixel_truth": False,
        }

    @classmethod
    def from_data(cls, value: object) -> "TypedAxisInventory":
        raw = _fields(
            value,
            {
                "schema",
                "algorithm_id",
                "matrix",
                "nominations",
                "formulas",
                "formula_count",
                "maximum_formula_count",
                "admitted_formula_ids",
                "empty_gap",
                "formula_language",
                "conjunction_precedence",
                "query_rows_seen",
                "model_calls_for_derivation_or_replay",
                "ranking_present",
                "negation_present",
                "lean_present",
                "calibrated_sets_claim_deterministic_pixel_truth",
            },
            "typed axis inventory",
        )
        if (
            raw["schema"] != INVENTORY_SCHEMA
            or raw["algorithm_id"] != ALGORITHM_ID
            or raw["maximum_formula_count"] != MAX_FORMULA_COUNT
            or raw["formula_language"] != "positive_equality_singleton_or_pair"
            or raw["conjunction_precedence"]
            != ["error", "certified_absent", "all_present", "indeterminate"]
            or raw["query_rows_seen"] != 0
            or raw["model_calls_for_derivation_or_replay"] != 0
            or raw["ranking_present"] is not False
            or raw["negation_present"] is not False
            or raw["lean_present"] is not False
            or raw["calibrated_sets_claim_deterministic_pixel_truth"] is not False
        ):
            raise TypedAxisSlateError("typed axis inventory policy differs")
        matrix = TypedSupportMatrix.from_data(raw["matrix"])
        nominations = TypedNominationSlate.from_data(raw["nominations"])
        rebuilt = cls.derive(matrix, nominations)
        if rebuilt.to_data() != dict(raw):
            raise TypedAxisSlateError("typed axis inventory deterministic replay differs")
        return rebuilt


def cold_replay_typed_axis_inventory(
    inventory: TypedAxisInventory, *, expected_inventory_address: str
) -> TypedAxisInventory:
    if type(inventory) is not TypedAxisInventory:
        raise TypeError("cold replay needs exact TypedAxisInventory")
    _address(expected_inventory_address, "expected inventory address")
    restored = TypedAxisInventory.from_data(inventory.to_data())
    if restored.inventory_address != expected_inventory_address:
        raise TypedAxisSlateError("typed axis inventory address differs")
    return restored


__all__ = (
    "ALGORITHM_ID",
    "AXES",
    "AXIS_DOMAINS",
    "Axis",
    "AxisNomination",
    "EvidenceKind",
    "EvidenceWitness",
    "EqualityAtom",
    "FormulaEvaluation",
    "FormulaRowEvaluation",
    "MAX_FORMULA_COUNT",
    "SupportSide",
    "TypedAxisCell",
    "TypedAxisInventory",
    "TypedAxisSlateError",
    "TypedEmptyGap",
    "TypedNominationSlate",
    "TypedSupportMatrix",
    "TypedSupportRow",
    "cold_replay_typed_axis_inventory",
)

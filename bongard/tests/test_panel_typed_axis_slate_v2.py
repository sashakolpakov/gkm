from __future__ import annotations

import copy

import pytest

from bongard.evidence import Disposition
from bongard.panel_typed_axis_slate_v2 import (
    AXES,
    Axis,
    AxisNomination,
    EqualityAtom,
    EvidenceKind,
    EvidenceWitness,
    FormulaRowEvaluation,
    MAX_FORMULA_COUNT,
    SupportSide,
    TypedAxisCell,
    TypedAxisInventory,
    TypedAxisSlateError,
    TypedNominationSlate,
    TypedSupportMatrix,
    TypedSupportRow,
    cold_replay_typed_axis_inventory,
)


PROTOCOL = "sha256:" + "a" * 64
GRANT = "sha256:" + "b" * 64

PRIMARY_VALUES = {
    Axis.TOPOLOGY: "closed",
    Axis.COMPONENT_COUNT: 1,
    Axis.STRAIGHT_ACTION_COUNT: 4,
    Axis.PRIMITIVE_MIX_OR_ARC_COUNT: "straight_only",
    Axis.TURNING_CONVEXITY: "convex_turning",
    Axis.SYMMETRY: "none",
    Axis.ASPECT_ORIENTATION: "compact",
    Axis.TEXTURE: "plain",
}


def _row(index: int, side: SupportSide) -> TypedSupportRow:
    values = dict(PRIMARY_VALUES)
    if side is SupportSide.CONTRAST:
        values[Axis.TOPOLOGY] = "open"
    return TypedSupportRow(
        f"{side.value}_{index:02d}",
        side,
        tuple(
            TypedAxisCell.python_exact(axis, values[axis], PROTOCOL)
            for axis in AXES
        ),
    )


def _matrix() -> TypedSupportMatrix:
    return TypedSupportMatrix.freeze(
        tuple(_row(index, SupportSide.PRIMARY) for index in range(6))
        + tuple(_row(index, SupportSide.CONTRAST) for index in range(6))
    )


def _slate(matrix: TypedSupportMatrix) -> TypedNominationSlate:
    return TypedNominationSlate.freeze(
        matrix,
        tuple(
            AxisNomination.nominate(axis, PRIMARY_VALUES[axis]) for axis in AXES
        ),
    )


def test_cell_equality_states_preserve_exact_vs_calibrated_authority() -> None:
    exact_match = TypedAxisCell.python_exact(Axis.TOPOLOGY, "closed", PROTOCOL)
    exact_excluded = TypedAxisCell.python_exact(Axis.TOPOLOGY, "open", PROTOCOL)
    calibrated_match = TypedAxisCell.calibrated_set(
        Axis.TOPOLOGY, ["closed"], PROTOCOL, GRANT
    )
    calibrated_wide = TypedAxisCell.calibrated_set(
        Axis.TOPOLOGY, ["open", "closed"], PROTOCOL, GRANT
    )
    calibrated_excluded = TypedAxisCell.calibrated_set(
        Axis.TOPOLOGY, ["open"], PROTOCOL, GRANT
    )
    gap = TypedAxisCell.gap(Axis.TOPOLOGY, PROTOCOL, "observer_unavailable")
    error = TypedAxisCell.error(Axis.TOPOLOGY, PROTOCOL, "malformed_output")

    assert exact_match.equality_disposition("closed") is Disposition.PRESENT
    assert exact_excluded.equality_disposition("closed") is Disposition.CERTIFIED_ABSENT
    assert calibrated_match.equality_disposition("closed") is Disposition.PRESENT
    assert calibrated_wide.equality_disposition("closed") is Disposition.INDETERMINATE
    assert calibrated_excluded.equality_disposition("closed") is Disposition.CERTIFIED_ABSENT
    assert gap.equality_disposition("closed") is Disposition.INDETERMINATE
    assert error.equality_disposition("closed") is Disposition.ERROR

    exact_witness = EvidenceWitness.evaluate(exact_excluded, "closed")
    calibrated_witness = EvidenceWitness.evaluate(calibrated_excluded, "closed")
    assert exact_witness.evidence_kind is EvidenceKind.PYTHON_EXACT
    assert exact_witness.basis_code == "python_exact_exclusion"
    assert exact_witness.deterministic_pixel_truth_claimed is True
    assert exact_witness.calibration_grant_address is None
    assert calibrated_witness.evidence_kind is EvidenceKind.CALIBRATED_SET
    assert calibrated_witness.basis_code == "calibrated_set_exclusion"
    assert calibrated_witness.deterministic_pixel_truth_claimed is False
    assert calibrated_witness.calibration_grant_address == GRANT


def test_conjunction_error_precedes_absence_then_indeterminate() -> None:
    base = _row(0, SupportSide.CONTRAST)
    cells = list(base.cells)
    cells[AXES.index(Axis.COMPONENT_COUNT)] = TypedAxisCell.error(
        Axis.COMPONENT_COUNT, PROTOCOL, "observer_crash"
    )
    row = TypedSupportRow(base.row_key, base.side, tuple(cells))
    evaluation = FormulaRowEvaluation.evaluate(
        "formula_00",
        (
            EqualityAtom(Axis.TOPOLOGY, "closed"),
            EqualityAtom(Axis.COMPONENT_COUNT, 1),
        ),
        row,
    )
    assert evaluation.atom_witnesses[0].disposition is Disposition.CERTIFIED_ABSENT
    assert evaluation.atom_witnesses[1].disposition is Disposition.ERROR
    assert evaluation.disposition is Disposition.ERROR
    assert [item.axis for item in evaluation.failure_witnesses] == [
        Axis.COMPONENT_COUNT
    ]

    cells[AXES.index(Axis.COMPONENT_COUNT)] = TypedAxisCell.gap(
        Axis.COMPONENT_COUNT, PROTOCOL, "observer_gap"
    )
    row = TypedSupportRow(base.row_key, base.side, tuple(cells))
    evaluation = FormulaRowEvaluation.evaluate(
        "formula_00",
        (
            EqualityAtom(Axis.TOPOLOGY, "closed"),
            EqualityAtom(Axis.COMPONENT_COUNT, 1),
        ),
        row,
    )
    assert evaluation.disposition is Disposition.CERTIFIED_ABSENT
    assert [item.axis for item in evaluation.failure_witnesses] == [Axis.TOPOLOGY]


def test_fixed_slate_enumerates_8_singletons_28_pairs_and_admits_support_formulae() -> None:
    matrix = _matrix()
    inventory = TypedAxisInventory.derive(matrix, _slate(matrix))

    assert len(inventory.formulas) == MAX_FORMULA_COUNT == 36
    assert [len(item.atoms) for item in inventory.formulas[:8]] == [1] * 8
    assert [len(item.atoms) for item in inventory.formulas[8:]] == [2] * 28
    assert inventory.admitted_formula_ids == (
        "formula_00",
        "formula_08",
        "formula_09",
        "formula_10",
        "formula_11",
        "formula_12",
        "formula_13",
        "formula_14",
    )
    assert inventory.empty_gap is None

    topology = inventory.formulas[0]
    assert topology.primary_counts == (6, 0, 0, 0)
    assert topology.contrast_counts == (0, 6, 0, 0)
    assert topology.admitted is True
    assert all(not row.failure_witnesses for row in topology.rows[:6])
    assert all(
        row.failure_witnesses[0].basis_code == "python_exact_exclusion"
        for row in topology.rows[6:]
    )

    component = inventory.formulas[1]
    assert component.primary_counts == (6, 0, 0, 0)
    assert component.contrast_counts == (6, 0, 0, 0)
    assert component.admitted is False
    assert "contrast_present_nonzero" in component.admission_failure_codes
    assert inventory.to_data()["query_rows_seen"] == 0
    assert inventory.to_data()["model_calls_for_derivation_or_replay"] == 0
    assert inventory.to_data()["lean_present"] is False


def test_nomination_gaps_reduce_search_and_empty_result_is_typed() -> None:
    matrix = _matrix()
    slate = TypedNominationSlate.freeze(
        matrix,
        tuple(AxisNomination.gap(axis, "axis_unavailable") for axis in AXES),
    )
    inventory = TypedAxisInventory.derive(matrix, slate)

    assert inventory.formulas == ()
    assert inventory.admitted_formula_ids == ()
    assert inventory.empty_gap is not None
    assert inventory.empty_gap.nomination_gap_axes == AXES
    assert inventory.empty_gap.evaluated_formula_count == 0
    assert inventory.empty_gap.rejected_formula_ids == ()


def test_round_trip_cold_replay_and_nested_tamper_detection() -> None:
    matrix = _matrix()
    inventory = TypedAxisInventory.derive(matrix, _slate(matrix))
    restored = TypedAxisInventory.from_data(inventory.to_data())
    assert restored == inventory
    assert (
        cold_replay_typed_axis_inventory(
            inventory, expected_inventory_address=inventory.inventory_address
        )
        == inventory
    )

    changed = copy.deepcopy(inventory.to_data())
    changed["formulas"][0]["rows"][6]["disposition"] = "present"
    with pytest.raises(TypedAxisSlateError):
        TypedAxisInventory.from_data(changed)

    changed = copy.deepcopy(inventory.to_data())
    changed["formulas"][0]["rows"][6]["atom_witnesses"][0][
        "evidence_kind"
    ] = "calibrated_set"
    with pytest.raises(TypedAxisSlateError):
        TypedAxisInventory.from_data(changed)

    changed = copy.deepcopy(inventory.to_data())
    changed["matrix"]["rows"][0]["cells"][0]["possible_values"] = ["open"]
    with pytest.raises(TypedAxisSlateError):
        TypedAxisInventory.from_data(changed)

    with pytest.raises(TypedAxisSlateError):
        cold_replay_typed_axis_inventory(
            inventory, expected_inventory_address="sha256:" + "f" * 64
        )


def test_matrix_and_nomination_shape_fail_closed() -> None:
    rows = tuple(_row(index, SupportSide.PRIMARY) for index in range(6)) + tuple(
        _row(index, SupportSide.CONTRAST) for index in range(6)
    )
    with pytest.raises(TypedAxisSlateError):
        TypedSupportMatrix.freeze(rows[:-1])
    with pytest.raises(TypedAxisSlateError):
        TypedSupportMatrix.freeze((rows[6],) + rows[1:6] + (rows[0],) + rows[7:])

    matrix = TypedSupportMatrix.freeze(rows)
    nominations = tuple(
        AxisNomination.nominate(axis, PRIMARY_VALUES[axis]) for axis in AXES
    )
    with pytest.raises(TypedAxisSlateError):
        TypedNominationSlate.freeze(matrix, nominations[:-1])
    with pytest.raises(TypedAxisSlateError):
        TypedNominationSlate.freeze(matrix, nominations[::-1])

    with pytest.raises(TypedAxisSlateError):
        TypedAxisCell.calibrated_set(
            Axis.TOPOLOGY, ["closed"], PROTOCOL, "not-an-address"
        )
    with pytest.raises(TypedAxisSlateError):
        TypedAxisCell.python_exact(Axis.COMPONENT_COUNT, True, PROTOCOL)


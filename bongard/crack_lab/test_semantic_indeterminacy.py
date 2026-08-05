"""Regressions for the semantic-absence/indeterminate/error boundary."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import Problem
from cofibered_proposer import ProposalBundle, _leg_lines
from semantic_compiler import (
    FailedValue,
    IndeterminateValue,
    compile_hypothesis,
)
from semantic_ir import DiagramEdge, DiagramSpec, LegCall, MorphSpec, SemanticHypothesis
import semantic_legs as L
from semantic_verifier import verify_hypothesis
from run_semantic_cone import _score_table, _status_of


def _uncertain_measure(_panel: np.ndarray) -> float:
    raise L.WitnessIndeterminate("poor_fit", "evidence does not resolve the fit")


def _identity_measure(value: float) -> float:
    return value


def _registry(*, declare: bool = True) -> L.LegRegistry:
    registry = L.default_registry()
    registry.register(L.LegContract(
        name="uncertain_measure",
        domain=("Panel",),
        codomain="Measurement",
        implementation=_uncertain_measure,
        indeterminate_modes=("poor_fit",) if declare else (),
        proxy_for=("uncertain",),
        measurement_kind="continuous",
    ))
    registry.register(L.LegContract(
        name="identity_measure",
        domain=("Measurement",),
        codomain="Measurement",
        implementation=_identity_measure,
        proxy_for=("uncertain",),
        measurement_kind="continuous",
    ))
    return registry


def _hypothesis() -> SemanticHypothesis:
    return SemanticHypothesis(
        version="0.1",
        hypothesis_id="indeterminate_is_not_negative",
        description="The panel is uncertain.",
        polarity="positive_satisfies",
        diagram=DiagramSpec((
            DiagramEdge(
                "raw", LegCall("uncertain_measure", ("panel",))),
            DiagramEdge(
                "score", LegCall("identity_measure", ("raw",))),
        )),
        score_node="score",
        order="high_positive",
        preservation_morphisms=(MorphSpec("translate", "panel"),),
        semantic_requirements=("uncertain",),
    )


def _problem() -> Problem:
    blank = np.zeros((24, 24), dtype=np.uint8)
    marked = blank.copy()
    marked[4:20, 12] = 1
    return Problem(
        "indeterminate_fixture", "fixture", "harness_only",
        tuple(marked.copy() for _ in range(6)),
        tuple(blank.copy() for _ in range(6)),
    )


def test_compiler_propagates_declared_indeterminacy_as_a_distinct_value() -> None:
    registry = _registry()
    cone = compile_hypothesis(_hypothesis(), registry)
    trace = cone.trace(np.zeros((24, 24), dtype=np.uint8), registry)

    assert isinstance(trace.node_values["raw"], IndeterminateValue)
    assert isinstance(trace.node_values["score"], IndeterminateValue)
    assert trace.leg_status == {
        "raw": "indeterminate",
        "score": "blocked_by_indeterminate",
    }
    assert trace.errors == ()
    assert trace.witness_absences == {}
    assert trace.witness_indeterminacies == {
        "raw": ("uncertain_measure", "poor_fit"),
    }


def test_undeclared_indeterminacy_is_an_implementation_error() -> None:
    registry = _registry(declare=False)
    cone = compile_hypothesis(_hypothesis(), registry)
    trace = cone.trace(np.zeros((24, 24), dtype=np.uint8), registry)

    assert isinstance(trace.node_values["raw"], FailedValue)
    assert trace.leg_status["raw"] == "error:UndeclaredWitnessIndeterminacy"
    assert "poor_fit" in trace.errors[0]
    assert trace.witness_indeterminacies == {}


def test_verifier_never_coerces_indeterminate_panels_to_negative() -> None:
    result = verify_hypothesis(_hypothesis(), _registry(), _problem())

    assert not result.accepted
    assert not result.semantic_admissible
    assert result.predicate_errors == 0
    assert result.structural_absences == 0
    assert result.indeterminate_evaluations == 12
    assert result.witness_absences == {}
    assert result.witness_indeterminacies == {
        "pos:raw:uncertain_measure:poor_fit": 6,
        "neg:raw:uncertain_measure:poor_fit": 6,
    }
    assert result.score_dispositions == ("indeterminate",) * 12
    assert result.support_predictions == (None,) * 12
    # Every unknown evaluation fails every empirical gate, including the six
    # negative panels that a boolean False coercion would have rewarded.
    assert result.support_errors == 12
    assert result.loo_errors == 12
    assert result.rotated_loo_errors == 72


def test_verifier_records_errors_separately_from_indeterminacy() -> None:
    result = verify_hypothesis(_hypothesis(), _registry(declare=False), _problem())

    assert result.score_dispositions == ("error",) * 12
    assert result.support_predictions == (None,) * 12
    assert result.predicate_errors == 12
    assert result.indeterminate_evaluations == 0
    assert result.structural_absences == 0


def test_runner_reports_indeterminate_instead_of_absent() -> None:
    result = verify_hypothesis(_hypothesis(), _registry(), _problem())
    bundle = ProposalBundle(
        problem_id="indeterminate_fixture",
        hypotheses=(_hypothesis(),),
        raw_text="",
        proposer_kind="fixture",
    )

    assert _status_of(result, bundle) == "INDETERMINATE"
    table = _score_table(result)
    assert table.count("INDETERMINATE") == 12
    assert "SEMANTIC_ABSENT" not in table


def test_registry_rejects_overlapping_absence_and_indeterminate_modes() -> None:
    contract = L.LegContract(
        name="ambiguous_disposition",
        domain=("Panel",),
        codomain="Measurement",
        implementation=lambda _panel: 0.0,
        failure_modes=("same_mode",),
        indeterminate_modes=("same_mode",),
        measurement_kind="continuous",
    )
    with pytest.raises(ValueError, match="both semantic absence and indeterminate"):
        L.LegRegistry().register(contract)


def test_part_graph_parse_failure_is_indeterminate_not_absence(monkeypatch) -> None:
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[5:27, 16] = 1
    obj = L.ObjectMask(mask, "stroke")

    def unresolved(_obj):
        raise L.WitnessAbsent("not_simple_curve", "cannot order contour")

    monkeypatch.setattr(L, "extract_contours", unresolved)
    with pytest.raises(L.WitnessIndeterminate) as failure:
        L.build_part_graph(obj)
    assert failure.value.failure_mode == "not_simple_curve"

    registry = L.default_registry()
    for name in (
            "decompose_component_into_parts", "build_part_graph",
            "build_object_part_graph"):
        contract = registry.get(name)
        assert "not_simple_curve" in contract.indeterminate_modes
        assert "not_simple_curve" not in contract.failure_modes


def test_proposer_registry_exposes_failure_dispositions() -> None:
    lines = _leg_lines().splitlines()
    angle = next(
        line for line in lines if line.startswith("- minimum_incident_angle:"))
    assert "[semantic absence: no_junction]" in angle
    assert "[indeterminate: insufficient_incident_rays, high_residual]" in angle

"""Support-only synthesis and hidden-query contracts."""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import grounded_predicate_ir as G
import grounded_synthesis as S


def _registry(*observable_ids: str) -> G.ObservableRegistry:
    registry = G.ObservableRegistry()
    for observable_id in observable_ids:
        registry.register(G.ObservableContract(
            observable_id=observable_id,
            value_type="real",
            unit="ratio",
            referent="fixture.panel",
            reducer="identity",
            evaluator=lambda context, key=observable_id: context[key],
            semantic_absence_modes=("no-referent",),
            indeterminate_modes=("ambiguous",),
        ))
    return registry


def _case(case_id: str, label: bool, **values) -> S.SupportCase:
    return S.SupportCase(case_id, values, label)


def _present(value: float, lower: float | None = None,
             upper: float | None = None) -> G.Present:
    return G.Present(value, "ratio", lower=lower, upper=upper)


def test_synthesis_finds_conjunction_when_no_atom_separates() -> None:
    registry = _registry("fixture/a", "fixture/b")
    intents = (
        S.MeasurementIntent("a-high", "fixture/a", "high"),
        S.MeasurementIntent("b-high", "fixture/b", "high"),
    )
    support = (
        _case("p0", True, **{"fixture/a": _present(10),
                             "fixture/b": _present(10)}),
        _case("p1", True, **{"fixture/a": _present(11),
                             "fixture/b": _present(11)}),
        _case("n0", False, **{"fixture/a": _present(2),
                              "fixture/b": _present(10)}),
        _case("n1", False, **{"fixture/a": _present(10),
                              "fixture/b": _present(2)}),
    )
    frozen = S.synthesize_grounded_predicate(intents, support, registry)
    assert isinstance(frozen.predicate, G.All)
    assert len(frozen.predicate.children) == 2
    assert frozen.selected_intent_ids == ("a-high", "b-high")
    assert frozen.support_evaluation.exact
    assert frozen.support_evaluation.predictions == \
        (True, True, False, False)
    assert all(not isinstance(child, G.Not)
               for child in frozen.predicate.children)


def test_wrong_polarity_is_rejected_not_negated() -> None:
    registry = _registry("fixture/value")
    support = (
        _case("p", True, **{"fixture/value": _present(10)}),
        _case("n", False, **{"fixture/value": _present(2)}),
    )
    with pytest.raises(S.NoGroundedSeparator):
        S.synthesize_grounded_predicate((
            S.MeasurementIntent("wrong-low", "fixture/value", "low"),
        ), support, registry)


def test_semantic_absence_is_negative_but_indeterminacy_blocks_atom() -> None:
    registry = _registry("fixture/value")
    intent = (S.MeasurementIntent("high", "fixture/value", "high"),)
    absent_support = (
        _case("p", True, **{"fixture/value": _present(10)}),
        _case("n", False, **{
            "fixture/value": G.SemanticAbsent("no-referent")}),
    )
    frozen = S.synthesize_grounded_predicate(
        intent, absent_support, registry)
    assert frozen.support_evaluation.exact

    unknown_support = (
        absent_support[0],
        _case("n", False, **{
            "fixture/value": G.Indeterminate("ambiguous")}),
    )
    with pytest.raises(S.NoGroundedSeparator):
        S.synthesize_grounded_predicate(intent, unknown_support, registry)


def test_interval_margin_is_frozen_before_hidden_query() -> None:
    registry = _registry("fixture/value")
    support = (
        _case("p0", True, **{
            "fixture/value": _present(9, lower=8, upper=10)}),
        _case("p1", True, **{
            "fixture/value": _present(10, lower=9, upper=11)}),
        _case("n", False, **{
            "fixture/value": _present(3, lower=2, upper=4)}),
    )
    frozen = S.synthesize_grounded_predicate((
        S.MeasurementIntent("high", "fixture/value", "high"),
    ), support, registry)
    assert isinstance(frozen.predicate, G.Compare)
    assert frozen.predicate.operator is G.ComparisonOperator.GT
    assert frozen.predicate.threshold.value == pytest.approx(6.0)

    hidden = (
        _case("hidden-p", True, **{
            "fixture/value": _present(8, lower=7, upper=9)}),
        _case("hidden-n", False, **{
            "fixture/value": _present(5, lower=4, upper=6)}),
    )
    evaluation = S.evaluate_hidden_queries(frozen, hidden)
    assert evaluation.exact
    assert evaluation.predictions == (True, False)
    assert frozen.predicate.threshold.value == pytest.approx(6.0)


def test_threshold_sensitivity_is_fixed_formula_diagnostic_not_loo() -> None:
    registry = _registry("fixture/value")
    cases = (
        _case("p", True, **{"fixture/value": _present(10)}),
        _case("n", False, **{"fixture/value": _present(2)}),
    )
    frozen = S.synthesize_grounded_predicate((
        S.MeasurementIntent("high", "fixture/value", "high"),
    ), cases, registry)
    points = S.threshold_sensitivity(
        frozen, cases, registry, relative_delta=0.05)
    assert [point.relative_delta for point in points] == [-0.05, 0.0, 0.05]
    assert points[1].predicate.to_dict() == frozen.predicate.to_dict()
    assert all(len(point.evaluation.cases) == len(cases) for point in points)

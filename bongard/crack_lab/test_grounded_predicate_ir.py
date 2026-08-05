"""Contract, type-safety, and four-valued laws for grounded predicate IR v0.2."""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import grounded_predicate_ir as G


def _contract(
    observable_id: str,
    value,
    *,
    value_type: G.ValueType = G.ValueType.INTEGER,
    unit: G.Unit = G.Unit.COUNT,
    absence_modes: tuple[str, ...] = (),
    indeterminate_modes: tuple[str, ...] = (),
    source: G.ObservableSource = G.ObservableSource.DETERMINISTIC,
    version: str = "v1",
) -> G.ObservableContract:
    return G.ObservableContract(
        observable_id=observable_id,
        value_type=value_type,
        unit=unit,
        referent="panel",
        reducer=G.Reducer.COUNT,
        evaluator=lambda _context: value,
        semantic_absence_modes=absence_modes,
        indeterminate_modes=indeterminate_modes,
        invariances=(
            G.Invariance.REFLECTION,
            G.Invariance.ROTATION,
            G.Invariance.TRANSLATION,
        ),
        source=source,
        version=version,
    )


def _registry(*contracts: G.ObservableContract) -> G.ObservableRegistry:
    registry = G.ObservableRegistry()
    for contract in contracts:
        registry.register(contract)
    return registry


def _comparison(observable_id: str, value: int = 2) -> G.Compare:
    return G.Compare(observable_id, G.ComparisonOperator.EQ, G.Literal(value, G.Unit.COUNT))


def _outcome_predicate(outcome: G.Observation, observable_id: str = "fixture/value"):
    registry = _registry(_contract(
        observable_id,
        outcome,
        absence_modes=("no-referent",),
        indeterminate_modes=("ambiguous",),
    ))
    return G.compile_predicate(_comparison(observable_id), registry)


def test_registered_present_observable_evaluates_comparison() -> None:
    compiled = _outcome_predicate(G.Present(2, G.Unit.COUNT))
    result = compiled.evaluate(object())
    assert result == G.Present(True, G.Unit.BOOLEAN)
    assert compiled.taint is G.Taint.PURE


@pytest.mark.parametrize("operator,threshold,upper,expected", [
    ("eq", 3, None, True),
    ("lt", 4, None, True),
    ("le", 3, None, True),
    ("gt", 2, None, True),
    ("ge", 3, None, True),
    ("between", 2, 4, True),
    ("between", 3, 3, True),
    ("between", 4, 7, False),
])
def test_all_comparison_operators(
    operator: str, threshold: int, upper: int | None, expected: bool,
) -> None:
    registry = _registry(_contract("fixture/count", G.Present(3, "count")))
    node = G.Compare(
        "fixture/count",
        operator,
        G.Literal(threshold, "count"),
        G.Literal(upper, "count") if upper is not None else None,
    )
    assert G.compile_predicate(node, registry).evaluate(None) == \
        G.Present(expected, G.Unit.BOOLEAN)


def test_indeterminate_is_preserved_under_negation() -> None:
    uncertain = _outcome_predicate(G.Indeterminate("ambiguous", "two parses"))
    uncertain_result = G.compile_predicate(
        G.Not(uncertain.predicate),
        _registry(_contract(
            "fixture/value",
            G.Indeterminate("ambiguous", "two parses"),
            indeterminate_modes=("ambiguous",),
        )),
    ).evaluate(None)

    assert isinstance(uncertain_result, G.Indeterminate)
    assert uncertain_result.mode == "ambiguous"


def test_semantic_absence_refutes_atom_and_negation_is_true() -> None:
    registry = _registry(_contract(
        "geometry/point-contact-gap-ratio",
        G.SemanticAbsent("no-point-contact", "certified by contact graph"),
        value_type=G.ValueType.REAL,
        unit=G.Unit.RATIO,
        absence_modes=("no-point-contact",),
    ))
    atom = G.Compare(
        "geometry/point-contact-gap-ratio", "lt", G.Literal(0.2, "ratio"))
    positive = G.compile_predicate(atom, registry).evaluate(None)
    negated_trace = G.compile_predicate(
        G.Not(atom), registry).evaluate_with_trace(None)
    assert positive == G.Present(False, G.Unit.BOOLEAN)
    assert negated_trace.result == G.Present(True, G.Unit.BOOLEAN)
    # The negative certificate is not erased by Boolean evaluation.
    assert isinstance(negated_trace.observations[0][1], G.SemanticAbsent)


def test_false_and_unknown_is_false() -> None:
    registry = _registry(
        _contract("fixture/false", G.Present(0, "count")),
        _contract(
            "fixture/unknown",
            G.Indeterminate("ambiguous"),
            indeterminate_modes=("ambiguous",),
        ),
    )
    node = G.All((
        _comparison("fixture/false", 2),
        _comparison("fixture/unknown", 2),
    ))
    assert G.compile_predicate(node, registry).evaluate(None) == \
        G.Present(False, G.Unit.BOOLEAN)


def test_true_and_unknown_is_unknown() -> None:
    registry = _registry(
        _contract("fixture/true", G.Present(2, "count")),
        _contract(
            "fixture/unknown",
            G.Indeterminate("ambiguous"),
            indeterminate_modes=("ambiguous",),
        ),
    )
    result = G.compile_predicate(G.All((
        _comparison("fixture/true"),
        _comparison("fixture/unknown"),
    )), registry).evaluate(None)
    assert isinstance(result, G.Indeterminate)


def test_true_or_unknown_is_true_and_false_or_unknown_is_unknown() -> None:
    unknown = G.Indeterminate("ambiguous")
    registry = _registry(
        _contract("fixture/true", G.Present(2, "count")),
        _contract("fixture/false", G.Present(0, "count")),
        _contract(
            "fixture/unknown", unknown, indeterminate_modes=("ambiguous",)),
    )
    true_or_unknown = G.Any((
        _comparison("fixture/true"), _comparison("fixture/unknown")))
    false_or_unknown = G.Any((
        _comparison("fixture/false"), _comparison("fixture/unknown")))
    assert G.compile_predicate(true_or_unknown, registry).evaluate(None) == \
        G.Present(True, G.Unit.BOOLEAN)
    assert isinstance(
        G.compile_predicate(false_or_unknown, registry).evaluate(None),
        G.Indeterminate,
    )


@pytest.mark.parametrize("combiner", [G.All, G.Any])
def test_error_is_never_masked_by_boolean_short_circuit(combiner) -> None:
    registry = _registry(
        _contract("fixture/false", G.Present(0, "count")),
        _contract("fixture/true", G.Present(2, "count")),
        _contract("fixture/error", G.Error("broken-observable")),
    )
    masking_value = _comparison(
        "fixture/false" if combiner is G.All else "fixture/true")
    node = combiner((masking_value, _comparison("fixture/error")))
    result = G.compile_predicate(node, registry).evaluate(None)
    assert isinstance(result, G.Error)
    assert result.code == "broken-observable"


def test_boolean_connectives_obey_strong_kleene_truth_tables() -> None:
    outcomes: dict[str, G.Observation] = {
        "t": G.Present(2, "count"),
        "f": G.Present(0, "count"),
        "u": G.Indeterminate("ambiguous"),
    }
    and_expected = {
        ("t", "t"): True, ("t", "f"): False, ("t", "u"): None,
        ("f", "t"): False, ("f", "f"): False, ("f", "u"): False,
        ("u", "t"): None, ("u", "f"): False, ("u", "u"): None,
    }
    or_expected = {
        ("t", "t"): True, ("t", "f"): True, ("t", "u"): True,
        ("f", "t"): True, ("f", "f"): False, ("f", "u"): None,
        ("u", "t"): True, ("u", "f"): None, ("u", "u"): None,
    }
    for left_name, left in outcomes.items():
        for right_name, right in outcomes.items():
            registry = _registry(
                _contract(
                    "fixture/left", left, indeterminate_modes=("ambiguous",)),
                _contract(
                    "fixture/right", right, indeterminate_modes=("ambiguous",)),
            )
            children = (_comparison("fixture/left"), _comparison("fixture/right"))
            for node, expected in (
                (G.All(children), and_expected[(left_name, right_name)]),
                (G.Any(children), or_expected[(left_name, right_name)]),
            ):
                result = G.compile_predicate(node, registry).evaluate(None)
                if expected is None:
                    assert isinstance(result, G.Indeterminate)
                else:
                    assert result == G.Present(expected, G.Unit.BOOLEAN)


def test_unknown_unit_is_rejected_before_evaluation() -> None:
    with pytest.raises(ValueError, match="unknown unit"):
        G.Literal(3, "mystery-units")
    with pytest.raises(ValueError, match="unknown observable contract enum value"):
        _contract("fixture/count", G.Present(2, "count"), unit="mystery-units")


def test_nominal_units_cannot_be_compared_even_when_values_are_numeric() -> None:
    registry = _registry(G.ObservableContract(
        observable_id="fixture/angle",
        value_type="real",
        unit="degrees",
        referent="panel/junction",
        reducer="min",
        evaluator=lambda _: G.Present(30.0, "degrees"),
    ))
    with pytest.raises(G.UnitMismatchError, match="radians.*degrees"):
        G.compile_predicate(
            G.Compare("fixture/angle", "lt", G.Literal(1.0, "radians")),
            registry,
        )


@pytest.mark.parametrize("value,value_type,unit", [
    (2.0, "integer", "count"),
    (1, "boolean", "boolean"),
    (True, "real", "ratio"),
    (3, "text", "text"),
])
def test_present_runtime_value_must_match_contract_type(value, value_type, unit) -> None:
    contract = G.ObservableContract(
        observable_id="fixture/value",
        value_type=value_type,
        unit=unit,
        referent="panel",
        reducer="identity",
        evaluator=lambda _: G.Present(value, unit),
    )
    result = contract.evaluate(None)
    assert isinstance(result, G.Error)
    assert result.code == "observable-contract-violation"


def test_order_comparison_rejects_boolean_and_text_observables() -> None:
    for value_type, unit, value, threshold in (
        ("boolean", "boolean", True, True),
        ("text", "text", "bird-like", "bird-like"),
    ):
        registry = _registry(G.ObservableContract(
            observable_id="oracle/label",
            value_type=value_type,
            unit=unit,
            referent="panel",
            reducer="identity",
            evaluator=lambda _, v=value, u=unit: G.Present(v, u),
        ))
        with pytest.raises(G.PredicateTypeError, match="requires a numeric observable"):
            G.compile_predicate(
                G.Compare("oracle/label", "gt", G.Literal(threshold, unit)),
                registry,
            )


def test_between_is_inclusive_and_rejects_reversed_bounds() -> None:
    registry = _registry(G.ObservableContract(
        observable_id="fixture/ratio",
        value_type="real",
        unit="ratio",
        referent="panel/contact",
        reducer="ratio",
        evaluator=lambda _: G.Present(0.25, "ratio"),
    ))
    at_boundary = G.Compare(
        "fixture/ratio", "between",
        G.Literal(0.25, "ratio"), G.Literal(0.5, "ratio"))
    assert G.compile_predicate(at_boundary, registry).evaluate(None) == \
        G.Present(True, G.Unit.BOOLEAN)
    with pytest.raises(G.PredicateCompileError, match="lower bound exceeds"):
        G.compile_predicate(G.Compare(
            "fixture/ratio", "between",
            G.Literal(0.5, "ratio"), G.Literal(0.25, "ratio")), registry)


@pytest.mark.parametrize("operator,threshold,lower,upper,expected", [
    ("lt", 30.0, 20.0, 25.0, True),
    ("lt", 30.0, 30.0, 35.0, False),
    ("le", 30.0, 20.0, 30.0, True),
    ("gt", 30.0, 31.0, 35.0, True),
    ("gt", 30.0, 20.0, 30.0, False),
    ("ge", 30.0, 30.0, 35.0, True),
    ("eq", 30.0, 30.0, 30.0, True),
    ("eq", 30.0, 31.0, 35.0, False),
])
def test_interval_comparison_decides_only_whole_interval(
    operator: str,
    threshold: float,
    lower: float,
    upper: float,
    expected: bool,
) -> None:
    registry = _registry(G.ObservableContract(
        "geometry/junction-angle", "real", "degrees", "panel/junction", "min",
        lambda _: G.Present(0.5 * (lower + upper), "degrees", (), lower, upper),
    ))
    node = G.Compare(
        "geometry/junction-angle", operator, G.Literal(threshold, "degrees"))
    assert G.compile_predicate(node, registry).evaluate(None) == \
        G.Present(expected, G.Unit.BOOLEAN)


@pytest.mark.parametrize("operator,threshold,lower,upper", [
    ("lt", 30.0, 25.0, 35.0),
    ("le", 30.0, 25.0, 35.0),
    ("gt", 30.0, 25.0, 35.0),
    ("ge", 30.0, 25.0, 35.0),
    ("eq", 30.0, 25.0, 35.0),
])
def test_boundary_overlap_is_indeterminate_and_stays_unknown_under_not(
    operator: str,
    threshold: float,
    lower: float,
    upper: float,
) -> None:
    registry = _registry(G.ObservableContract(
        "geometry/junction-angle", "real", "degrees", "panel/junction", "min",
        lambda _: G.Present(
            0.5 * (lower + upper), "degrees", ("angle-fit",), lower, upper),
    ))
    atom = G.Compare(
        "geometry/junction-angle", operator, G.Literal(threshold, "degrees"))
    for predicate in (atom, G.Not(atom)):
        result = G.compile_predicate(predicate, registry).evaluate(None)
        assert isinstance(result, G.Indeterminate)
        assert result.mode == "comparison-boundary-overlap"
        assert result.provenance == ("angle-fit",)


def test_between_interval_can_satisfy_violate_or_overlap() -> None:
    def result_for(lower: float, upper: float) -> G.Observation:
        registry = _registry(G.ObservableContract(
            "geometry/gap-ratio", "real", "ratio", "panel/contact", "ratio",
            lambda _: G.Present(
                0.5 * (lower + upper), "ratio", (), lower, upper),
        ))
        node = G.Compare(
            "geometry/gap-ratio", "between",
            G.Literal(0.2, "ratio"), G.Literal(0.4, "ratio"))
        return G.compile_predicate(node, registry).evaluate(None)

    assert result_for(0.25, 0.35) == G.Present(True, G.Unit.BOOLEAN)
    assert result_for(0.45, 0.55) == G.Present(False, G.Unit.BOOLEAN)
    assert isinstance(result_for(0.1, 0.3), G.Indeterminate)


def test_present_interval_is_finite_ordered_numeric_only() -> None:
    with pytest.raises(ValueError, match="finite number"):
        G.Present(2.0, "ratio", (), float("nan"), 3.0)
    with pytest.raises(ValueError, match="lower <= value <= upper"):
        G.Present(2.0, "ratio", (), 2.5, 3.0)
    with pytest.raises(ValueError, match="only to numeric"):
        G.Present("bird-like", "text", (), 0.0, 1.0)


def test_compiler_rejects_unregistered_observable() -> None:
    with pytest.raises(G.UnknownObservableError, match="unknown observable"):
        G.compile_predicate(_comparison("proposer/invented"), G.ObservableRegistry())


def test_oracle_leaf_taints_entire_predicate_hybrid() -> None:
    pure = _contract("geometry/part-count", G.Present(2, "count"))
    oracle = G.ObservableContract(
        observable_id="oracle/bird-likeness",
        value_type="real",
        unit="ratio",
        referent="panel",
        reducer="identity",
        evaluator=lambda _: G.Present(0.9, "ratio"),
        indeterminate_modes=("ambiguous-rubric",),
        source=G.ObservableSource.ORACLE,
    )
    registry = _registry(pure, oracle)
    pure_compiled = G.compile_predicate(
        _comparison("geometry/part-count"), registry)
    hybrid = G.compile_predicate(G.All((
        _comparison("geometry/part-count"),
        G.Compare("oracle/bird-likeness", "ge", G.Literal(0.7, "ratio")),
    )), registry)
    assert pure_compiled.taint is G.Taint.PURE
    assert hybrid.taint is G.Taint.HYBRID
    assert hybrid.canonical_dict()["taint"] == "HYBRID"


def test_contract_rejects_undeclared_absence_and_indeterminate_modes() -> None:
    absent = _contract(
        "fixture/absent", G.SemanticAbsent("no-contact"),
        absence_modes=("no-object",))
    uncertain = _contract(
        "fixture/uncertain", G.Indeterminate("unstable-fit"),
        indeterminate_modes=("occluded",))
    for contract in (absent, uncertain):
        result = contract.evaluate(None)
        assert isinstance(result, G.Error)
        assert result.code == "observable-contract-violation"


def test_evaluator_exception_and_non_outcome_fail_closed() -> None:
    def explode(_):
        raise RuntimeError("boom")

    exploding = _contract("fixture/explode", G.Present(2, "count"))
    object.__setattr__(exploding, "evaluator", explode)
    raw = _contract("fixture/raw", G.Present(2, "count"))
    object.__setattr__(raw, "evaluator", lambda _: -1)
    assert isinstance(exploding.evaluate(None), G.Error)
    raw_result = raw.evaluate(None)
    assert isinstance(raw_result, G.Error)
    assert "must return" in raw_result.detail


def test_contract_version_digest_binds_semantics_not_callable_identity() -> None:
    first = _contract(
        "geometry/part-count", G.Present(2, "count"), version="v1")
    same = _contract(
        "geometry/part-count", G.Present(99, "count"), version="v1")
    changed_version = _contract(
        "geometry/part-count", G.Present(2, "count"), version="v2")
    changed_unit = G.ObservableContract(
        observable_id="geometry/part-count",
        value_type="real",
        unit="ratio",
        referent="panel",
        reducer="ratio",
        evaluator=lambda _: G.Present(0.2, "ratio"),
        version="v1",
        invariances=("translation", "rotation", "reflection"),
    )
    assert first.version_digest() == same.version_digest()
    assert first.version_digest() != changed_version.version_digest()
    assert first.version_digest() != changed_unit.version_digest()


def test_registry_digest_is_insertion_order_independent_and_rejects_duplicates() -> None:
    left = _contract("geometry/part-count", G.Present(2, "count"))
    right = _contract("geometry/contact-count", G.Present(1, "count"))
    first = _registry(left, right)
    second = _registry(right, left)
    assert first.version_digest() == second.version_digest()
    with pytest.raises(ValueError, match="already registered"):
        first.register(left)


def test_canonical_predicate_digest_is_commutative_for_all_and_any() -> None:
    registry = _registry(
        _contract("geometry/part-count", G.Present(2, "count")),
        _contract("geometry/contact-count", G.Present(1, "count")),
    )
    a = _comparison("geometry/part-count", 2)
    b = _comparison("geometry/contact-count", 1)
    assert G.compile_predicate(G.All((a, b)), registry).digest == \
        G.compile_predicate(G.All((b, a)), registry).digest
    assert G.compile_predicate(G.Any((a, b)), registry).digest == \
        G.compile_predicate(G.Any((b, a)), registry).digest


def test_json_round_trip_is_canonical_and_strict() -> None:
    node = G.Not(G.All((
        _comparison("geometry/part-count", 2),
        G.Compare(
            "geometry/gap-ratio", "between",
            G.Literal(0.1, "ratio"), G.Literal(0.3, "ratio")),
    )))
    encoded = node.to_dict()
    decoded = G.predicate_from_dict(encoded)
    assert decoded.to_dict() == encoded
    assert G.canonical_digest(decoded.to_dict()) == G.canonical_digest(encoded)
    with pytest.raises(ValueError, match="unknown fields"):
        G.predicate_from_dict({
            **_comparison("geometry/part-count").to_dict(),
            "python": "arbitrary-code",
        })


def test_evaluation_trace_caches_repeated_observable_and_binds_receipt() -> None:
    calls = 0

    def evaluate(_):
        nonlocal calls
        calls += 1
        return G.Present(2, "count", ("fixture-receipt",))

    registry = _registry(G.ObservableContract(
        observable_id="geometry/part-count",
        value_type="integer",
        unit="count",
        referent="panel",
        reducer="count",
        evaluator=evaluate,
    ))
    compiled = G.compile_predicate(G.All((
        G.Compare("geometry/part-count", "ge", G.Literal(1, "count")),
        G.Compare("geometry/part-count", "le", G.Literal(3, "count")),
    )), registry)
    trace = compiled.evaluate_with_trace(object())
    assert calls == 1
    assert trace.result == G.Present(True, G.Unit.BOOLEAN)
    assert len(trace.observations) == 1
    assert trace.to_dict()["predicate_digest"] == compiled.digest


def test_adapter_shapes_for_part_contact_and_gap_ratio_are_closed_world() -> None:
    """Exercise the three first integration observables against a plain context."""
    panel = {"parts": 2, "contacts": 1, "point_contact_gap_ratio": 0.18}
    registry = G.ObservableRegistry()
    registry.register(G.ObservableContract(
        "geometry/part-count", "integer", "count", "panel/part-graph", "count",
        lambda context: G.Present(context["parts"], "count"),
        invariances=("translation", "rotation", "reflection", "uniform-scale"),
    ))
    registry.register(G.ObservableContract(
        "geometry/contact-count", "integer", "count", "panel/part-graph", "count",
        lambda context: G.Present(context["contacts"], "count"),
        invariances=("translation", "rotation", "reflection", "uniform-scale"),
    ))
    registry.register(G.ObservableContract(
        "geometry/point-contact-gap-ratio", "real", "ratio",
        "panel/point-contact", "ratio",
        lambda context: G.Present(context["point_contact_gap_ratio"], "ratio"),
        semantic_absence_modes=("no-point-contact",),
        indeterminate_modes=("ambiguous-contact",),
        invariances=("translation", "rotation", "reflection", "uniform-scale"),
    ))
    predicate = G.All((
        G.Compare("geometry/part-count", "eq", G.Literal(2, "count")),
        G.Compare("geometry/contact-count", "eq", G.Literal(1, "count")),
        G.Compare(
            "geometry/point-contact-gap-ratio", "between",
            G.Literal(0.1, "ratio"), G.Literal(0.25, "ratio")),
    ))
    compiled = G.compile_predicate(predicate, registry)
    assert compiled.taint is G.Taint.PURE
    assert compiled.evaluate(panel) == G.Present(True, G.Unit.BOOLEAN)

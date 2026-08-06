from __future__ import annotations

import pytest

from bongard.evidence import Disposition, Evidence, Provenance, SoftSemanticObservation, Uncertainty
from bongard.ir import (
    AllOf,
    AnyOf,
    Atom,
    IRValidationError,
    Interval,
    Quantity,
    Relation,
    StaticLegCall,
    evaluate_formula,
    formula_from_data,
    validate_formula,
)
from bongard.legs import (
    AffirmativeRelation,
    BOOLEAN_WITNESS,
    PANEL,
    SOFT_SEMANTIC,
    InvarianceContract,
    LegContract,
    LegRegistry,
    Transform,
    TypedValue,
    Unit,
    ValueType,
)


ANGLE = ValueType("measurement", Unit.DEGREES)


def provenance(method: str) -> Provenance:
    return Provenance("test-vision", "1", method, input_digests=("panel",))


def angle_leg(panel: dict[str, object]) -> Evidence[float]:
    if panel.get("crash"):
        raise RuntimeError("fit crashed")
    if panel.get("absent"):
        return Evidence.certified_absent(provenance("angle"), "no line segments")
    bounds = panel["angle"]
    assert isinstance(bounds, tuple)
    lower, upper = bounds
    return Evidence.present(
        (float(lower) + float(upper)) / 2.0,
        provenance("angle"),
        Uncertainty(float(lower), float(upper)),
    )


def bird_leg(panel: dict[str, object]) -> Evidence[SoftSemanticObservation]:
    lower, upper = panel["bird"]  # type: ignore[misc]
    origin = provenance("bird-description")
    observation = SoftSemanticObservation(
        "bird-like object", Uncertainty(lower, upper), origin
    )
    return Evidence.present(observation, origin)


def registry() -> tuple[LegRegistry, object, object]:
    legs = LegRegistry()
    angle_ref = legs.register(
        LegContract(
            "oblique_angle",
            "1",
            (PANEL,),
            ANGLE,
            angle_leg,
            affirmative_relations=frozenset(
                {AffirmativeRelation.AT_LEAST, AffirmativeRelation.AT_MOST}
            ),
            invariance=InvarianceContract(
                invariant_under=frozenset({Transform.TRANSLATION})
            ),
        )
    )
    bird_ref = legs.register(
        LegContract(
            "bird_like",
            "1",
            (PANEL,),
            SOFT_SEMANTIC,
            bird_leg,
            affirmative_relations=frozenset({AffirmativeRelation.AT_LEAST}),
        )
    )
    return legs.freeze(), angle_ref, bird_ref


def angle_atom(reference: object, relation: Relation, threshold: float) -> Atom:
    return Atom(
        StaticLegCall(reference, ("panel",)),  # type: ignore[arg-type]
        relation,
        "object has a calibrated oblique angle",
        Quantity(threshold, Unit.DEGREES),
    )


@pytest.mark.parametrize(
    ("bounds", "expected"),
    [
        ((61.0, 64.0), Disposition.PRESENT),
        ((40.0, 49.0), Disposition.CERTIFIED_ABSENT),
        ((49.0, 52.0), Disposition.INDETERMINATE),
    ],
)
def test_interval_comparison_never_uses_midpoint(bounds, expected) -> None:
    legs, angle_ref, _ = registry()
    formula = angle_atom(angle_ref, Relation.AT_LEAST, 50.0)
    result = evaluate_formula(
        formula, legs, {"panel": TypedValue(PANEL, {"angle": bounds})}
    )
    assert result.disposition is expected


def test_soft_observation_only_becomes_decision_through_calibrated_interval() -> None:
    legs, _, bird_ref = registry()
    formula = Atom(
        StaticLegCall(bird_ref, ("panel",)),
        Relation.AT_LEAST,
        "calibrated bird-like support",
        Quantity(0.7, Unit.PROBABILITY),
    )
    present = evaluate_formula(
        formula, legs, {"panel": TypedValue(PANEL, {"bird": (0.75, 0.83)})}
    )
    unresolved = evaluate_formula(
        formula, legs, {"panel": TypedValue(PANEL, {"bird": (0.65, 0.80)})}
    )
    assert present.disposition is Disposition.PRESENT
    assert unresolved.disposition is Disposition.INDETERMINATE


def test_extractor_crash_and_absence_remain_different() -> None:
    legs, angle_ref, _ = registry()
    formula = angle_atom(angle_ref, Relation.AT_LEAST, 50.0)
    crashed = evaluate_formula(
        formula, legs, {"panel": TypedValue(PANEL, {"crash": True})}
    )
    absent = evaluate_formula(
        formula, legs, {"panel": TypedValue(PANEL, {"absent": True})}
    )
    assert crashed.disposition is Disposition.ERROR
    assert absent.disposition is Disposition.CERTIFIED_ABSENT


def test_false_payload_cannot_smuggle_negation_through_present_witness() -> None:
    def malformed_boolean_witness(_: object) -> Evidence[bool]:
        return Evidence.present(False, provenance("malformed-boolean"))

    legs = LegRegistry()
    reference = legs.register(
        LegContract(
            "malformed_boolean",
            "1",
            (PANEL,),
            BOOLEAN_WITNESS,
            malformed_boolean_witness,
        )
    )
    legs.freeze()
    formula = Atom(
        StaticLegCall(reference, ("panel",)),
        Relation.PRESENT,
        "an affirmative witness exists",
    )
    result = evaluate_formula(
        formula, legs, {"panel": TypedValue(PANEL, {})}
    )
    assert result.disposition is Disposition.ERROR
    assert "affirmative witness True" in (result.reason or "")


def test_static_typecheck_rejects_wrong_units_and_unknown_boundary() -> None:
    legs, angle_ref, _ = registry()
    wrong_unit = Atom(
        StaticLegCall(angle_ref, ("panel",)),
        Relation.AT_LEAST,
        "wrongly unitless",
        Quantity(0.5, Unit.PROBABILITY),
    )
    with pytest.raises(IRValidationError, match="threshold uses probability"):
        validate_formula(wrong_unit, legs, {"panel": PANEL})
    with pytest.raises(IRValidationError, match="unknown boundary"):
        validate_formula(
            angle_atom(angle_ref, Relation.AT_LEAST, 50), legs, {"image": PANEL}
        )


def test_scalar_cannot_win_by_using_an_undeclared_opposite_orientation() -> None:
    legs = LegRegistry()
    reference = legs.register(
        LegContract(
            "affirmative_angle",
            "1",
            (PANEL,),
            ANGLE,
            angle_leg,
            affirmative_relations=frozenset({AffirmativeRelation.AT_LEAST}),
        )
    )
    legs.freeze()
    flipped = angle_atom(reference, Relation.AT_MOST, 50.0)
    with pytest.raises(IRValidationError, match="not an affirmative orientation"):
        validate_formula(flipped, legs, {"panel": PANEL})


def test_closed_parser_has_no_not_or_polarity_escape_hatch() -> None:
    with pytest.raises(IRValidationError, match="atom/and/or only"):
        formula_from_data({"type": "not", "term": {}})
    with pytest.raises(IRValidationError, match="unknown polarity"):
        formula_from_data(
            {
                "type": "and",
                "terms": [{}, {}],
                "justification": "x",
                "polarity": "negative_wins",
            }
        )


def test_positive_and_or_have_explicit_justification_and_sound_dispositions() -> None:
    legs, angle_ref, _ = registry()
    high = angle_atom(angle_ref, Relation.AT_LEAST, 50.0)
    low = angle_atom(angle_ref, Relation.AT_MOST, 70.0)
    conjunction = AllOf((high, low), "the same angle lies in a broad oblique band")
    disjunction = AnyOf((high, low), "either calibrated oblique prototype")
    panel = {"panel": TypedValue(PANEL, {"angle": (55.0, 60.0)})}
    assert evaluate_formula(conjunction, legs, panel).disposition is Disposition.PRESENT
    assert evaluate_formula(disjunction, legs, panel).disposition is Disposition.PRESENT
    with pytest.raises(ValueError, match="justification"):
        AllOf((high, low), "")


def test_interval_value_unit_is_checked_at_runtime() -> None:
    def bad_interval(_: object) -> Evidence[Interval]:
        return Evidence.present(
            Interval(0.1, 0.2, Unit.PROBABILITY), provenance("bad-unit")
        )

    legs = LegRegistry()
    ref = legs.register(
        LegContract(
            "bad_angle",
            "1",
            (PANEL,),
            ANGLE,
            bad_interval,
            affirmative_relations=frozenset({AffirmativeRelation.AT_LEAST}),
        )
    )
    legs.freeze()
    result = evaluate_formula(
        angle_atom(ref, Relation.AT_LEAST, 10),
        legs,
        {"panel": TypedValue(PANEL, {})},
    )
    assert result.disposition is Disposition.ERROR
    assert "unit differs" in (result.reason or "")


def test_scalar_point_must_lie_inside_its_declared_uncertainty() -> None:
    def contradictory_angle(_: object) -> Evidence[float]:
        return Evidence.present(
            90.0,
            provenance("contradictory-angle"),
            Uncertainty(10.0, 20.0),
        )

    legs = LegRegistry()
    reference = legs.register(
        LegContract(
            "contradictory_angle",
            "1",
            (PANEL,),
            ANGLE,
            contradictory_angle,
            affirmative_relations=frozenset({AffirmativeRelation.AT_LEAST}),
        )
    )
    legs.freeze()
    result = evaluate_formula(
        angle_atom(reference, Relation.AT_LEAST, 50.0),
        legs,
        {"panel": TypedValue(PANEL, {})},
    )
    assert result.disposition is Disposition.ERROR
    assert "lies outside its uncertainty interval" in (result.reason or "")

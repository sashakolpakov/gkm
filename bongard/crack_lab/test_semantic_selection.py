import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from semantic_selection import (
    RISK_FIELDS,
    CandidateEvaluation,
    ComplexityBreakdown,
    RiskVector,
    Track,
    UnmeasuredRiskError,
    admissibility_issues,
    complexity_for_cone,
    conditional_free_energy,
    pareto_frontier,
    rank_candidates,
    select_candidate,
)


def _risk(**overrides):
    values = {name: 0.0 for name in RISK_FIELDS}
    values.update(overrides)
    return RiskVector(**values)


def _candidate(candidate_id, risk, complexity=0, *, admissible=True,
               lambda_value=0.02):
    return CandidateEvaluation(
        candidate_id,
        Track.SEMANTIC_PURE,
        admissible,
        risk,
        ComplexityBreakdown(diagram_node_cost=complexity),
        lambda_value=lambda_value,
    )


def test_unmeasured_risks_are_explicit_and_strict_by_default():
    risk = RiskVector(R_support=0.1, R_rotated_LOO=0.2)

    assert risk.measured_fields == ("R_support", "R_rotated_LOO")
    assert "R_contrast" in risk.unmeasured_fields
    with pytest.raises(UnmeasuredRiskError) as exc:
        risk.scalar()
    assert "R_contrast" in exc.value.risk_fields

    # A partial protocol is legal only when the caller names it.
    assert risk.scalar(risk_fields=("R_support", "R_rotated_LOO")) == pytest.approx(0.3)
    assert risk.scalar(unmeasured="exclude") == pytest.approx(0.3)
    assert risk.scalar(
        unmeasured="penalize", unmeasured_penalty=0.5) == pytest.approx(2.8)


def test_partial_candidate_serialization_never_reports_unknown_as_zero():
    candidate = _candidate(
        "partial", RiskVector(R_support=0.0, R_rotated_LOO=0.0))

    data = candidate.to_dict()
    assert data["risk"]["R_contrast"] is None
    assert "R_contrast" in data["unmeasured_risks"]
    assert data["free_energy"] is None
    with pytest.raises(UnmeasuredRiskError):
        _ = candidate.free_energy


def test_conditional_free_energy_uses_only_an_explicit_protocol():
    risk = RiskVector(R_support=0.1, R_rotated_LOO=0.2)
    complexity = ComplexityBreakdown(diagram_node_cost=5)

    score = conditional_free_energy(
        risk,
        complexity,
        0.1,
        risk_fields=("R_support", "R_rotated_LOO"),
    )
    assert score == pytest.approx(0.8)


def test_pareto_uses_every_measured_risk_dimension():
    cheap_but_bad_contrast = _candidate(
        "cheap",
        _risk(R_support=0.0, R_rotated_LOO=0.0, R_contrast=0.8),
        complexity=1,
    )
    expensive_but_good_contrast = _candidate(
        "semantic",
        _risk(R_support=0.0, R_rotated_LOO=0.0, R_contrast=0.0),
        complexity=2,
    )
    dominated = _candidate(
        "dominated",
        _risk(R_support=0.2, R_rotated_LOO=0.2, R_contrast=0.9),
        complexity=3,
    )

    frontier = pareto_frontier(
        [cheap_but_bad_contrast, expensive_but_good_contrast, dominated])
    assert [candidate.candidate_id for candidate in frontier] == ["cheap", "semantic"]


def test_pareto_excludes_semantically_inadmissible_candidates():
    invalid = _candidate("invalid", _risk(), complexity=0, admissible=False)
    valid = _candidate("valid", _risk(R_support=0.1), complexity=3)

    assert pareto_frontier([invalid, valid]) == [valid]
    assert admissibility_issues(invalid) == ("semantic_inadmissible",)


def test_partial_vectors_with_different_measurement_signatures_are_incomparable():
    support_only = _candidate("support", RiskVector(R_support=0.0), complexity=1)
    support_and_contrast = _candidate(
        "contrast", RiskVector(R_support=0.0, R_contrast=0.0), complexity=2)

    assert pareto_frontier([support_only, support_and_contrast]) == [
        support_only, support_and_contrast]
    assert pareto_frontier(
        [support_only, support_and_contrast],
        risk_fields=("R_support", "R_contrast"),
    ) == [support_and_contrast]


def test_required_risks_and_limits_are_admissibility_gates():
    partial = _candidate("partial", RiskVector(R_support=0.0))
    risky = _candidate("risky", _risk(R_support=0.25))
    safe = _candidate("safe", _risk(R_support=0.05))

    assert admissibility_issues(
        partial, required_risks=("R_support", "R_contrast")) == (
            "unmeasured:R_contrast",)
    assert pareto_frontier(
        [partial, risky, safe],
        required_risks=RISK_FIELDS,
        risk_limits={"R_support": 0.1},
    ) == [safe]


def test_runner_helpers_rank_only_measured_admissible_candidates():
    partial = _candidate("partial", RiskVector(R_support=0.0), complexity=0)
    simple = _candidate("simple", _risk(R_support=0.1), complexity=1)
    accurate = _candidate("accurate", _risk(R_support=0.0), complexity=10)
    invalid = _candidate("invalid", _risk(), complexity=0, admissible=False)

    ranked = rank_candidates(
        [partial, simple, accurate, invalid],
        risk_fields=RISK_FIELDS,
        pareto_only=False,
    )
    assert [candidate.candidate_id for candidate in ranked] == ["simple", "accurate"]
    assert select_candidate(
        [partial, simple, accurate, invalid],
        risk_fields=RISK_FIELDS,
        pareto_only=False,
    ) is simple


def test_candidate_selection_refuses_to_pool_different_tracks():
    semantic = _candidate("semantic", _risk())
    unrestricted = CandidateEvaluation(
        "unrestricted",
        Track.UNRESTRICTED,
        True,
        _risk(),
        ComplexityBreakdown(),
    )
    with pytest.raises(ValueError, match="cannot mix experiment tracks"):
        pareto_frontier([semantic, unrestricted])
    with pytest.raises(ValueError, match="cannot mix experiment tracks"):
        rank_candidates([semantic, unrestricted])


def test_complexity_charges_definitions_once_and_separates_namespaces():
    def edge(name, parameter_count=0):
        return SimpleNamespace(call=SimpleNamespace(
            leg_name=name,
            parameters=tuple((f"p{i}", i) for i in range(parameter_count)),
        ))

    cone = SimpleNamespace(
        used_legs=("parse_scene", "measure", "measure"),
        node_types={
            "panel": "Panel",
            "scene": "Scene",
            "w1": "ContourWitness",
            "w2": "ContourWitness",
        },
        hypothesis=SimpleNamespace(
            diagram=SimpleNamespace(edges=(
                edge("parse_scene"), edge("measure", 2), edge("measure"))),
            cofibrations=(SimpleNamespace(attachment_leg="attach"),),
        ),
    )

    complexity = complexity_for_cone(
        cone,
        promoted_legs={"parse_scene", "ContourWitness"},
        promoted_witness_types=set(),
        leg_definition_costs={"measure": 7},
        residual_code_cost=2,
        exception_cost=3,
        literal_lookup_cost=4,
    )
    # Repeated calls do not re-charge the definition; a witness type is not a
    # promoted leg merely because its string was placed in promoted_legs.
    assert complexity.new_leg_cost == 7
    assert complexity.witness_type_cost == 1
    assert complexity.diagram_node_cost == 3
    assert complexity.diagram_edge_cost == 3
    assert complexity.leg_call_cost == 3
    assert complexity.binding_cost == 3
    assert complexity.parameter_cost == 2
    assert complexity.cofibration_attachment_cost == 1
    assert complexity.total == 32

    promoted = complexity_for_cone(
        cone,
        promoted_legs={"parse_scene", "measure"},
        promoted_witness_types={"ContourWitness"},
    )
    assert promoted.new_leg_cost == 0
    assert promoted.witness_type_cost == 0


def test_invalid_risk_complexity_and_lambda_values_are_rejected():
    with pytest.raises(ValueError):
        RiskVector(R_support=float("nan"))
    with pytest.raises(ValueError):
        ComplexityBreakdown(new_leg_cost=-1)
    with pytest.raises(TypeError):
        ComplexityBreakdown(new_leg_cost=1.5)
    with pytest.raises(ValueError):
        _candidate("bad-lambda", _risk(), lambda_value=-0.1)
    with pytest.raises(KeyError):
        _risk().scalar(weights={"not_a_risk": 1.0})

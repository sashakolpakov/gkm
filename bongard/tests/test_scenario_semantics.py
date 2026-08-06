from __future__ import annotations

import pytest

from bongard.evidence import Disposition, Evidence, Provenance
from bongard.scenario_semantics import (
    ScenarioSemanticsError,
    evaluate_joint_scenario_conjunction,
)


def _provenance(name: str) -> Provenance:
    return Provenance("fixture", "1", name)


def _yes(name: str) -> Evidence[bool]:
    return Evidence.present(True, _provenance(name))


def _no(name: str) -> Evidence[bool]:
    return Evidence.certified_absent(_provenance(name), f"{name} mismatch")


def _unknown(name: str) -> Evidence[bool]:
    return Evidence.indeterminate(_provenance(name), f"{name} ambiguous")


def _error(name: str) -> Evidence[bool]:
    return Evidence.error(_provenance(name), "FixtureError", f"{name} failed")


def test_complete_conjunction_precedes_consensus_for_multimodal_near_misses() -> None:
    result = evaluate_joint_scenario_conjunction(
        {
            "balanced": {"component_count": _no("b-a"), "hole_count": _yes("b-b")},
            "permissive": {"component_count": _yes("p-a"), "hole_count": _no("p-b")},
            "strict": {"component_count": _no("s-a"), "hole_count": _yes("s-b")},
        }
    )
    # Neither atom has a scenario-wise absence consensus.  Nevertheless the
    # complete positive conjunction is constructively false in every scenario.
    assert result.evidence.disposition is Disposition.CERTIFIED_ABSENT
    assert dict(result.scenario_dispositions) == {
        "balanced": Disposition.CERTIFIED_ABSENT,
        "permissive": Disposition.CERTIFIED_ABSENT,
        "strict": Disposition.CERTIFIED_ABSENT,
    }


def test_scenario_disagreement_is_indeterminate_not_negative() -> None:
    result = evaluate_joint_scenario_conjunction(
        {
            "balanced": {"component_count": _yes("balanced")},
            "permissive": {"component_count": _no("permissive")},
            "strict": {"component_count": _yes("strict")},
        }
    )
    assert result.evidence.disposition is Disposition.INDETERMINATE


def test_any_extraction_or_atom_error_dominates_negative_evidence() -> None:
    result = evaluate_joint_scenario_conjunction(
        {
            "balanced": {"component_count": _no("balanced")},
            "permissive": {"component_count": _error("permissive")},
            "strict": {"component_count": _no("strict")},
        }
    )
    assert result.evidence.disposition is Disposition.ERROR
    assert result.evidence.error_type == "FixtureError"


def test_indeterminate_scenario_prevents_absence_consensus() -> None:
    result = evaluate_joint_scenario_conjunction(
        {
            "balanced": {"component_count": _no("balanced")},
            "permissive": {"component_count": _unknown("permissive")},
            "strict": {"component_count": _no("strict")},
        }
    )
    assert result.evidence.disposition is Disposition.INDETERMINATE


def test_joint_contract_requires_same_nonempty_atom_set() -> None:
    with pytest.raises(ScenarioSemanticsError, match="same complete"):
        evaluate_joint_scenario_conjunction(
            {
                "balanced": {"component_count": _yes("balanced")},
                "strict": {"hole_count": _yes("strict")},
            }
        )


def test_present_evidence_must_carry_true() -> None:
    malformed = object.__new__(Evidence)
    object.__setattr__(malformed, "disposition", Disposition.PRESENT)
    object.__setattr__(malformed, "provenance", _provenance("malformed"))
    object.__setattr__(malformed, "value", False)
    object.__setattr__(malformed, "uncertainty", None)
    object.__setattr__(malformed, "certificate", None)
    object.__setattr__(malformed, "reason", None)
    object.__setattr__(malformed, "error_type", None)
    with pytest.raises(ScenarioSemanticsError, match="must contain true"):
        evaluate_joint_scenario_conjunction(
            {
                "balanced": {"component_count": malformed},
                "strict": {"component_count": _yes("strict")},
            }
        )

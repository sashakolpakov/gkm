from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from bongard.evidence import Disposition, Evidence, Provenance
from bongard.ir import evaluate_formula
from bongard.legs import PANEL, TypedValue
from bongard.proposer import (
    HYBRID_EPISTEMIC_STATUS,
    HybridObservation,
    parse_rule_proposal,
)
from bongard.synthesis import compile_hybrid_proposal, truth_from_hybrid_observation


@dataclass(frozen=True)
class Receipt:
    receipt_digest: str = "receipt"
    input_digest: str = "input"
    thread_id: str = "thread"
    requested_model: str = "gpt-test"
    requested_reasoning_effort: str = "medium"

    def to_dict(self):
        return self.__dict__


def proposal():
    payload: dict[str, Any] = {
        "positive_description": "a bird-like object with a beak and two wings",
        "panel_descriptions": {
            **{f"pos_{i}": "bird-like silhouette" for i in range(6)},
            **{f"neg_{i}": "non-bird shape" for i in range(6)},
        },
        "view": "carrier_shape",
        "observable_requests": [],
        "formula_template": {"kind": "all", "atoms": ["hybrid_claim"]},
        "hybrid_claim": {
            "epistemic_status": HYBRID_EPISTEMIC_STATUS,
            "phrase": "bird-like object",
            "operational_definition": "one central body, two lateral wings, and a beak-like protrusion",
            "required_visual_cues": [
                {
                    "cue_id": "central_body",
                    "positive_description": "one central body",
                },
                {
                    "cue_id": "lateral_wings",
                    "positive_description": "two lateral wing-like lobes",
                },
                {
                    "cue_id": "beak_protrusion",
                    "positive_description": "one forward beak-like protrusion",
                },
            ],
        },
        "confidence": "medium",
    }
    return parse_rule_proposal(
        payload, receipt=Receipt(), observable_catalog={}  # type: ignore[arg-type]
    )


def observation(rule, disposition: Disposition) -> HybridObservation:
    origin = Provenance("observer", "1", "single-panel", (rule.digest,))
    if disposition is Disposition.PRESENT:
        evidence = Evidence.present(
            ("central_body", "lateral_wings", "beak_protrusion"), origin
        )
    elif disposition is Disposition.CERTIFIED_ABSENT:
        evidence = Evidence.certified_absent(origin, "fully visible shape has no wings")
    elif disposition is Disposition.INDETERMINATE:
        evidence = Evidence.indeterminate(origin, "ambiguous protrusions")
    else:
        evidence = Evidence.error(origin, "DecodeError", "image unreadable")
    return HybridObservation(rule.digest, {}, evidence, Receipt())  # type: ignore[arg-type]


@pytest.mark.parametrize("disposition", tuple(Disposition))
def test_compile_hybrid_preserves_every_observer_disposition(disposition) -> None:
    rule = proposal()

    def observer(candidate, panel):
        assert candidate == rule
        assert panel == "opaque-panel"
        return observation(rule, disposition)

    compiled = compile_hybrid_proposal(rule, observer=observer)
    result = evaluate_formula(
        compiled.formula,
        compiled.registry,
        {"panel": TypedValue(PANEL, "opaque-panel")},
    )
    assert result.disposition is disposition
    assert compiled.proposer_digest == rule.digest.removeprefix("sha256:")


def test_projection_never_turns_failure_into_absence() -> None:
    rule = proposal()
    failed = truth_from_hybrid_observation(observation(rule, Disposition.ERROR))
    unknown = truth_from_hybrid_observation(
        observation(rule, Disposition.INDETERMINATE)
    )
    assert failed.disposition is Disposition.ERROR
    assert unknown.disposition is Disposition.INDETERMINATE
    assert failed.disposition is not Disposition.CERTIFIED_ABSENT

from __future__ import annotations

from dataclasses import dataclass
import hashlib
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
from bongard.support_prototypes import (
    SUPPORT_PROTOTYPE_FEATURES,
    FeatureDimension,
    FeatureInterval,
    FrozenFeatureSpace,
    FrozenPanelFeatures,
    PositivePrototypeFormula,
    SupportPrototypePlan,
    fit_support_prototypes,
    panel_side_assignment_digest,
)
from bongard.synthesis import (
    SynthesisError,
    compile_hybrid_proposal,
    compile_prototype_proposal,
    truth_from_hybrid_observation,
)


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


def pure_proposal():
    payload: dict[str, Any] = {
        "positive_description": "shapes with a compact component layout",
        "panel_descriptions": {
            **{f"pos_{i}": "compact component layout" for i in range(6)},
            **{f"neg_{i}": "spread component layout" for i in range(6)},
        },
        "view": "carrier_shape",
        "observable_requests": [
            {
                "observable_id": "prototype.topology",
                "affirmative_interpretation": "a compact component layout is present",
                "arguments": {},
            }
        ],
        "formula_template": {"kind": "all", "atoms": ["prototype.topology"]},
        "hybrid_claim": None,
        "confidence": "medium",
    }
    return parse_rule_proposal(
        payload,
        receipt=Receipt(),  # type: ignore[arg-type]
        observable_catalog={"prototype.topology": "component topology"},
    )


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def prototype_inputs(rule):
    space = FrozenFeatureSpace(
        extractor_id="neutral-raster-features",
        extractor_version="1",
        extractor_artifact_digest=_digest("extractor"),
        preprocessing_digest=_digest("preprocessing"),
        receipt_protocol_digest=_digest("receipt"),
        dimensions=(FeatureDimension("compactness", "fraction", 0, 1, 1),),
    )

    def packet(name: str, value: float) -> FrozenPanelFeatures:
        return FrozenPanelFeatures(
            panel_digest=_digest("panel:" + name),
            feature_space_digest=space.digest(),
            extractor_receipt_digest=_digest("receipt:" + name),
            values=(FeatureInterval("compactness", value, value),),
        )

    positive = (packet("positive-a", 0.8), packet("positive-b", 0.9))
    negative = (packet("negative-a", 0.1), packet("negative-b", 0.2))
    plan = SupportPrototypePlan(
        space.digest(),
        panel_side_assignment_digest(
            tuple(item.panel_digest for item in positive),
            tuple(item.panel_digest for item in negative),
        ),
        2,
    )
    prototypes = fit_support_prototypes(
        plan,
        space,
        positive,
        negative,
        expected_plan_digest=plan.digest(),
    )
    predicate = PositivePrototypeFormula(
        claim="fixed positive-support prototype match for visual proposal "
        + rule.digest,
        feature_space_digest=space.digest(),
        prototype_digest=prototypes.digest(),
        support_assignment_digest=prototypes.support_assignment_digest,
        decision_margin=0.1,
    )
    return space, prototypes, predicate, packet("query", 0.85)


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


def test_compile_pure_proposal_uses_honest_prototype_boundary() -> None:
    rule = pure_proposal()
    space, prototypes, predicate, query = prototype_inputs(rule)
    compiled = compile_prototype_proposal(rule, space, prototypes, predicate)

    assert dict(compiled.attachment_contract.boundary_types) == {
        "features": SUPPORT_PROTOTYPE_FEATURES
    }
    assert compiled.formula.claim == (
        "fixed positive-support prototype match for visual proposal " + rule.digest
    )
    assert "compact component" not in compiled.formula.claim
    result = evaluate_formula(
        compiled.formula,
        compiled.registry,
        {"features": TypedValue(SUPPORT_PROTOTYPE_FEATURES, query)},
    )
    assert result.disposition is Disposition.PRESENT
    assert compiled.proposer_digest == rule.digest.removeprefix("sha256:")


def test_prototype_compiler_rejects_prose_overclaim_and_hybrid_input() -> None:
    rule = pure_proposal()
    space, prototypes, predicate, _ = prototype_inputs(rule)
    overstated = PositivePrototypeFormula(
        claim="the prose concept is mathematically proved",
        feature_space_digest=predicate.feature_space_digest,
        prototype_digest=predicate.prototype_digest,
        support_assignment_digest=predicate.support_assignment_digest,
        decision_margin=predicate.decision_margin,
    )
    with pytest.raises(SynthesisError, match="canonical operational claim"):
        compile_prototype_proposal(rule, space, prototypes, overstated)
    with pytest.raises(SynthesisError, match="PURE proposal"):
        compile_prototype_proposal(proposal(), space, prototypes, predicate)

from __future__ import annotations

import hashlib

from bongard.family_soft_leg import register_family_soft_predicate
from bongard.ir import Relation, evaluate_formula, validate_formula
from bongard.legs import FROZEN_VISUAL_SCORE, LegRegistry, TypedValue, Unit
from bongard.soft_predicates import (
    BlindSoftScoreRecord,
    SoftFamilyDevelopmentUnit,
    SoftScorerFamily,
    SoftScorerProtocol,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _protocol() -> SoftScorerProtocol:
    return SoftScorerProtocol(
        family_id="fixture-family",
        version="1",
        proposer_grammar_id="typed-visual-v1",
        proposer_grammar_digest=_digest("grammar"),
        proposer_model_id="fixture-proposer",
        proposer_reasoning_effort="medium",
        proposer_prompt_id="fixture-proposer-prompt",
        proposer_prompt_digest=_digest("proposer-prompt"),
        scorer_model_id="fixture-scorer",
        scorer_reasoning_effort="medium",
        scorer_prompt_template_id="fixture-scorer-prompt-template",
        scorer_prompt_template_digest=_digest("scorer-prompt-template"),
        scorer_decoder_id="ordinal-cues-v1",
        scorer_decoder_digest=_digest("decoder"),
        ordinal_map=(
            ("supported", 1.0),
            ("ambiguous", 0.5),
            ("unsupported", 0.0),
        ),
        aggregation="min",
        witness_extractor_id="fixture-witnesses",
        witness_extractor_digest=_digest("witness-extractor"),
        support_gate_id="visual-semantic-support-replay/v1",
        support_gate_digest=_digest("support-gate"),
        score_bin_edges=(0.0, 0.5, 1.0),
        affirmative_boundary=0.5,
        confidence_level=0.1,
        minimum_clusters_per_bin=8,
    )


def _family() -> SoftScorerFamily:
    protocol = _protocol()
    units = []
    for score, positive, prefix in ((0.0, False, "low"), (1.0, True, "high")):
        for index in range(8):
            units.append(
                SoftFamilyDevelopmentUnit(
                    observation_id=f"{prefix}-{index:02d}",
                    task_id=f"task-{prefix}-{index:02d}",
                    panel_digest=_digest(f"panel-{prefix}-{index}"),
                    claim_digest=_digest(f"claim-{prefix}-{index}"),
                    scorer_protocol_digest=protocol.digest(),
                    proposer_call_id=f"proposer-{prefix}-{index:02d}",
                    scorer_call_id=f"scorer-{prefix}-{index:02d}",
                    dependence_cluster_id=f"cluster-{prefix}-{index:02d}",
                    score_record_digest=_digest(f"score-{prefix}-{index}"),
                    annotation_receipt_digest=_digest(
                        f"annotation-{prefix}-{index}"
                    ),
                    score=score,
                    affirmative_label=positive,
                    score_bin_index=0 if score == 0.0 else 1,
                )
            )
    return SoftScorerFamily.fit(
        protocol,
        tuple(sorted(units, key=lambda item: item.observation_id)),
        expected_protocol_digest=protocol.digest(),
    )


def _record(
    family: SoftScorerFamily,
    claim_digest: str,
    judgment: str,
) -> BlindSoftScoreRecord:
    witnesses = [] if judgment == "unsupported" else ["component-00"]
    return BlindSoftScoreRecord.from_model_output(
        {
            "cue_judgments": [
                {
                    "cue_id": "cue-00",
                    "judgment": judgment,
                    "witness_ids": witnesses,
                }
            ]
        },
        scorer_protocol_digest=family.protocol_digest,
        task_id="fresh-task",
        panel_id="neutral-panel",
        panel_digest=_digest("fresh-panel-" + judgment),
        claim_digest=claim_digest,
        proposer_call_id="fresh-proposer-call",
        proposer_receipt_digest=_digest("fresh-proposer-receipt"),
        scorer_call_id="fresh-scorer-call-" + judgment,
        scorer_receipt_digest=_digest("fresh-scorer-receipt-" + judgment),
        witness_packet_digest=_digest("fresh-witness-packet"),
        pre_observation_commitment_digest=_digest(
            "frozen-proposal-policy-commitment"
        ),
        declared_cue_ids=("cue-00",),
        verifier_witness_ids=("component-00",),
    )


def test_registered_family_soft_leg_exposes_threshold_in_closed_ir() -> None:
    family = _family()
    claim_digest = _digest("fresh-bird-rubric")
    registry = LegRegistry()
    handle = register_family_soft_predicate(
        registry,
        name="family_soft_claim",
        version="1",
        family=family,
        expected_protocol_digest=family.protocol_digest,
        expected_family_digest=family.digest(),
        claim_digest=claim_digest,
        claim_description="bird-like articulated organization",
        cue_ids=("cue-00",),
    )
    registry.freeze()
    atom = handle.atom()
    assert atom.relation is Relation.AT_LEAST
    assert atom.lower is not None
    assert atom.lower.value == family.affirmative_boundary
    assert atom.lower.unit is Unit.PROBABILITY
    validate_formula(atom, registry, {"soft_score": FROZEN_VISUAL_SCORE})

    present = evaluate_formula(
        atom,
        registry,
        {
            "soft_score": TypedValue(
                FROZEN_VISUAL_SCORE,
                _record(family, claim_digest, "supported"),
            )
        },
    )
    absent = evaluate_formula(
        atom,
        registry,
        {
            "soft_score": TypedValue(
                FROZEN_VISUAL_SCORE,
                _record(family, claim_digest, "unsupported"),
            )
        },
    )
    assert present.disposition.value == "present"
    assert absent.disposition.value == "certified_absent"
    assert "operational_digest" in registry.snapshot().to_data()[0]


def test_family_soft_leg_rejects_score_for_another_claim_as_error() -> None:
    family = _family()
    claim_digest = _digest("expected-rubric")
    registry = LegRegistry()
    handle = register_family_soft_predicate(
        registry,
        name="family_soft_claim",
        version="1",
        family=family,
        expected_protocol_digest=family.protocol_digest,
        expected_family_digest=family.digest(),
        claim_digest=claim_digest,
        claim_description="one articulated animal-like carrier",
        cue_ids=("cue-00",),
    )
    registry.freeze()
    result = evaluate_formula(
        handle.atom(),
        registry,
        {
            "soft_score": TypedValue(
                FROZEN_VISUAL_SCORE,
                _record(family, _digest("different-rubric"), "supported"),
            )
        },
    )
    assert result.disposition.value == "error"
    assert result.error_type == "SoftPredicateIntegrityError"

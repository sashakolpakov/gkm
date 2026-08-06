from __future__ import annotations

from dataclasses import replace
import hashlib
import json

import pytest

from bongard.semantic_policy import (
    VisualSemanticPolicy,
    VisualSemanticPolicyError,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _policy() -> VisualSemanticPolicy:
    return VisualSemanticPolicy(
        witness_extractor_id="joint_visual_witnesses",
        witness_extractor_version="1",
        witness_extractor_digest=_digest("extractor"),
        witness_catalog_digest=_digest("witness-catalog"),
        scenario_ids=("balanced", "permissive", "strict"),
        direct_predicate_catalog_digest=_digest("direct-catalog"),
        proposal_protocol_digest=_digest("proposal-protocol"),
        soft_scorer_protocol_digest=_digest("soft-protocol"),
        soft_scorer_family_digest=_digest("soft-family"),
        soft_family_development_manifest_digest=_digest("soft-development"),
    )


def test_policy_round_trip_binds_python_first_semantics_without_backend_identity() -> None:
    policy = _policy()
    decoded = VisualSemanticPolicy.from_data(policy.to_data())
    assert decoded == policy
    assert decoded.digest() == policy.digest()
    encoded = json.dumps(policy.to_data(), sort_keys=True).lower()
    assert "complete_direct_conjunction_inside_each_scenario" in encoded
    assert '"task_local_threshold_fitting": false' in encoded
    assert '"model_emits_final_boolean": false' in encoded
    assert '"polarity_flip_allowed": false' in encoded
    assert "backend" not in encoded
    assert '"lean"' not in encoded
    assert "proof_assistant" not in encoded


def test_policy_identity_changes_with_every_executable_dependency() -> None:
    policy = _policy()
    variants = (
        replace(policy, witness_extractor_digest=_digest("changed extractor")),
        replace(policy, witness_catalog_digest=_digest("changed witnesses")),
        replace(
            policy,
            direct_predicate_catalog_digest=_digest("changed direct catalog"),
        ),
        replace(policy, proposal_protocol_digest=_digest("changed proposal")),
        replace(
            policy,
            soft_scorer_protocol_digest=_digest("changed soft protocol"),
        ),
        replace(policy, soft_scorer_family_digest=_digest("changed family")),
        replace(
            policy,
            soft_family_development_manifest_digest=_digest("changed development"),
        ),
    )
    assert len({policy.digest(), *(item.digest() for item in variants)}) == 8


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"polarity_flip_allowed": True}, "polarity_flip_allowed"),
        ({"negation_node_available": True}, "negation_node_available"),
        ({"reference_semantics": "some-proof-assistant/v1"}, "reference_semantics"),
    ],
)
def test_policy_decoder_rejects_semantic_drift(change, message) -> None:
    data = _policy().to_data()
    data.update(change)
    with pytest.raises(VisualSemanticPolicyError, match=message):
        VisualSemanticPolicy.from_data(data)


def test_policy_rejects_collapsed_or_duplicate_scenarios() -> None:
    policy = _policy()
    with pytest.raises(VisualSemanticPolicyError, match="at least two"):
        replace(policy, scenario_ids=("collapsed",))
    with pytest.raises(VisualSemanticPolicyError, match="sorted"):
        replace(policy, scenario_ids=("strict", "balanced"))
    with pytest.raises(VisualSemanticPolicyError, match="unique"):
        replace(policy, scenario_ids=("strict", "strict"))


def test_policy_rejects_support_fitted_capacity_changes() -> None:
    policy = _policy()
    with pytest.raises(VisualSemanticPolicyError, match="three direct"):
        replace(policy, max_direct_atoms=4)
    with pytest.raises(VisualSemanticPolicyError, match="one soft"):
        replace(policy, max_soft_claims=2)

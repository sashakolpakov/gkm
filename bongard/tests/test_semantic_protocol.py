from __future__ import annotations

from dataclasses import replace
import hashlib
import json

import pytest

from bongard.benchmark import SupportGatePolicy
from bongard.blind_soft_transport import (
    BLIND_SOFT_DECODER_ID,
    BLIND_SOFT_PROMPT_TEMPLATE_ID,
    blind_soft_decoder_digest,
    blind_soft_prompt_template_digest,
)
import bongard.semantic_protocol as semantic_protocol_module
from bongard.semantic_protocol import (
    SemanticProtocolIntegrityError,
    VISUAL_SOFT_WITNESS_INTERFACE_ID,
    build_prospective_soft_scorer_protocol,
    build_visual_semantic_policy,
    visual_semantic_proposal_procedure_data,
    visual_semantic_proposal_procedure_digest,
)
from bongard.soft_predicates import (
    SoftFamilyDevelopmentUnit,
    SoftScorerFamily,
    SoftScorerProtocol,
)
from bongard.typed_visual_proposal import (
    TYPED_VISUAL_PROPOSER_GRAMMAR_ID,
    TYPED_VISUAL_PROPOSER_PROMPT_ID,
    RegisteredAtomCatalog,
    typed_visual_proposal_grammar_digest,
    typed_visual_proposal_prompt_digest,
)
from bongard.visual_predicate_catalog import DIRECT_VISUAL_ATOM_CATALOG
from bongard.visual_witness_summaries import (
    visual_joint_soft_witness_interface_digest,
)
from bongard.visual_witness_bundle import (
    VISUAL_WITNESS_BUNDLE_EXTRACTOR_ID,
    VISUAL_WITNESS_BUNDLE_VERSION,
    visual_witness_bundle_catalog_digest,
    visual_witness_bundle_extractor_digest,
)
from bongard.visual_witnesses import (
    VISUAL_WITNESS_SCENARIO_IDS,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _protocol(**changes: object) -> SoftScorerProtocol:
    arguments: dict[str, object] = {
        "proposer_model_id": "gpt-proposer",
        "proposer_reasoning_effort": "medium",
        "scorer_model_id": "gpt-scorer",
        "scorer_reasoning_effort": "high",
        "score_bin_edges": (0.0, 0.25, 0.75, 1.0),
        "affirmative_boundary": 0.7,
        "confidence_level": 0.8,
        "minimum_clusters_per_bin": 2,
    }
    arguments.update(changes)
    return build_prospective_soft_scorer_protocol(**arguments)  # type: ignore[arg-type]


def _development_units(
    protocol: SoftScorerProtocol,
) -> tuple[SoftFamilyDevelopmentUnit, ...]:
    result: list[SoftFamilyDevelopmentUnit] = []
    levels = ((0.0, False), (0.5, True), (1.0, True))
    for bin_index, (score, label) in enumerate(levels):
        for cluster_index in range(2):
            key = f"bin-{bin_index}-cluster-{cluster_index}"
            result.append(
                SoftFamilyDevelopmentUnit(
                    observation_id=key,
                    task_id=f"task-{key}",
                    panel_digest=_digest(f"panel-{key}"),
                    claim_digest=_digest(f"claim-{key}"),
                    scorer_protocol_digest=protocol.digest(),
                    proposer_call_id=f"proposer-{key}",
                    scorer_call_id=f"scorer-{key}",
                    dependence_cluster_id=f"cluster-{key}",
                    score_record_digest=_digest(f"score-{key}"),
                    annotation_receipt_digest=_digest(f"annotation-{key}"),
                    score=score,
                    affirmative_label=label,
                    score_bin_index=bin_index,
                )
            )
    return tuple(result)


def _family(protocol: SoftScorerProtocol) -> SoftScorerFamily:
    return SoftScorerFamily.fit(
        protocol,
        _development_units(protocol),
        expected_protocol_digest=protocol.digest(),
    )


def test_prospective_protocol_exists_before_any_development_record() -> None:
    protocol = _protocol()
    gate = SupportGatePolicy.visual_semantic()

    assert protocol.proposer_grammar_id == TYPED_VISUAL_PROPOSER_GRAMMAR_ID
    assert protocol.proposer_grammar_digest == typed_visual_proposal_grammar_digest(
        DIRECT_VISUAL_ATOM_CATALOG
    )
    assert protocol.proposer_prompt_id == TYPED_VISUAL_PROPOSER_PROMPT_ID
    assert protocol.proposer_prompt_digest == typed_visual_proposal_prompt_digest(
        DIRECT_VISUAL_ATOM_CATALOG
    )
    assert protocol.scorer_prompt_template_id == BLIND_SOFT_PROMPT_TEMPLATE_ID
    assert protocol.scorer_prompt_template_digest == blind_soft_prompt_template_digest()
    assert protocol.scorer_decoder_id == BLIND_SOFT_DECODER_ID
    assert protocol.scorer_decoder_digest == blind_soft_decoder_digest()
    assert protocol.witness_extractor_id == VISUAL_SOFT_WITNESS_INTERFACE_ID
    assert protocol.witness_extractor_digest == (
        visual_joint_soft_witness_interface_digest()
    )
    assert protocol.support_gate_id == f"{gate.mode.value}@{gate.version}"
    assert protocol.support_gate_digest == semantic_protocol_module.canonical_digest(
        gate.to_data()
    )
    data = protocol.to_data()
    encoded = json.dumps(data, sort_keys=True).lower()
    assert "development_manifest" not in data
    assert "development_units" not in data
    assert "calibrated_support_intervals" not in encoded
    assert "family_digest" not in encoded
    assert "policy_digest" not in encoded


def test_every_prospective_executable_dependency_changes_protocol_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = _protocol().digest()
    assert _protocol(proposer_model_id="other-proposer").digest() != baseline
    assert _protocol(proposer_reasoning_effort="low").digest() != baseline
    assert _protocol(scorer_model_id="other-scorer").digest() != baseline
    assert _protocol(scorer_reasoning_effort="xhigh").digest() != baseline
    assert _protocol(score_bin_edges=(0.0, 0.5, 1.0)).digest() != baseline
    assert _protocol(affirmative_boundary=0.8).digest() != baseline
    assert _protocol(confidence_level=0.9).digest() != baseline
    assert _protocol(minimum_clusters_per_bin=3).digest() != baseline

    no_holes_catalog = RegisteredAtomCatalog(
        (DIRECT_VISUAL_ATOM_CATALOG.atoms[0],)
    )
    with monkeypatch.context() as patch:
        patch.setattr(
            semantic_protocol_module,
            "DIRECT_VISUAL_ATOM_CATALOG",
            no_holes_catalog,
        )
        assert _protocol().digest() != baseline

    digest_dependencies = (
        "typed_visual_proposal_grammar_digest",
        "typed_visual_proposal_prompt_digest",
        "blind_soft_prompt_template_digest",
        "blind_soft_decoder_digest",
        "visual_joint_soft_witness_interface_digest",
    )
    for name in digest_dependencies:
        with monkeypatch.context() as patch:
            replacement = lambda *args: "f" * 64
            patch.setattr(semantic_protocol_module, name, replacement)
            assert _protocol().digest() != baseline

    with monkeypatch.context() as patch:
        patch.setattr(
            semantic_protocol_module,
            "_visual_semantic_support_gate_identity",
            lambda: ("changed-gate", "e" * 64),
        )
        assert _protocol().digest() != baseline


def test_postfit_policy_tracks_visual_procedure_family_and_manifest_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = _protocol()
    family = _family(protocol)
    baseline = build_visual_semantic_policy(
        family,
        prospective_protocol=protocol,
    )

    dependencies = (
        "visual_witness_bundle_extractor_digest",
        "visual_witness_bundle_catalog_digest",
        "_proposal_transport_source_digest",
    )
    for name in dependencies:
        with monkeypatch.context() as patch:
            patch.setattr(semantic_protocol_module, name, lambda: "f" * 64)
            changed = build_visual_semantic_policy(
                family,
                prospective_protocol=protocol,
            )
            assert changed.digest() != baseline.digest()

    units = list(_development_units(protocol))
    units[0] = replace(
        units[0],
        annotation_receipt_digest=_digest("changed-annotation-receipt"),
    )
    changed_family = SoftScorerFamily.fit(
        protocol,
        tuple(units),
        expected_protocol_digest=protocol.digest(),
    )
    changed_policy = build_visual_semantic_policy(
        changed_family,
        prospective_protocol=protocol,
    )
    assert changed_family.development_manifest_digest != (
        family.development_manifest_digest
    )
    assert changed_family.digest() != family.digest()
    assert changed_policy.digest() != baseline.digest()


def test_protocol_family_policy_form_an_acyclic_identity_chain() -> None:
    protocol = _protocol()
    prospective_digest = protocol.digest()
    procedure = visual_semantic_proposal_procedure_data(protocol)
    family = _family(protocol)
    policy = build_visual_semantic_policy(
        family,
        prospective_protocol=protocol,
    )

    assert protocol.digest() == prospective_digest
    assert procedure["scorer_protocol_digest"] == prospective_digest
    assert procedure["family_digest_is_input"] is False
    assert procedure["visual_semantic_policy_digest_is_input"] is False
    assert family.protocol_digest == prospective_digest
    assert policy.soft_scorer_protocol_digest == prospective_digest
    assert policy.soft_scorer_family_digest == family.digest()
    assert (
        policy.soft_family_development_manifest_digest
        == family.development_manifest_digest
    )
    assert policy.proposal_protocol_digest == (
        visual_semantic_proposal_procedure_digest(protocol)
    )
    assert policy.witness_extractor_id == VISUAL_WITNESS_BUNDLE_EXTRACTOR_ID
    assert policy.witness_extractor_version == VISUAL_WITNESS_BUNDLE_VERSION
    assert policy.witness_extractor_digest == visual_witness_bundle_extractor_digest()
    assert policy.witness_catalog_digest == visual_witness_bundle_catalog_digest()
    assert policy.scenario_ids == VISUAL_WITNESS_SCENARIO_IDS
    assert policy.direct_predicate_catalog_digest == (
        DIRECT_VISUAL_ATOM_CATALOG.digest
    )
    assert "development_manifest" not in protocol.to_data()
    assert "visual_semantic_policy" not in family.to_data()


def test_family_protocol_mismatch_and_stale_dependencies_fail() -> None:
    protocol = _protocol()
    family = _family(protocol)
    other = _protocol(scorer_model_id="different-scorer")
    with pytest.raises(SemanticProtocolIntegrityError, match="does not belong"):
        build_visual_semantic_policy(
            family,
            prospective_protocol=other,
        )

    stale = replace(
        protocol,
        witness_extractor_digest="0" * 64,
    )
    with pytest.raises(
        SemanticProtocolIntegrityError,
        match="witness_extractor_digest",
    ):
        visual_semantic_proposal_procedure_digest(stale)


def test_all_protocol_and_policy_identities_are_checker_neutral() -> None:
    protocol = _protocol()
    family = _family(protocol)
    policy = build_visual_semantic_policy(
        family,
        prospective_protocol=protocol,
    )
    serialized = json.dumps(
        {
            "protocol": protocol.to_data(),
            "procedure": visual_semantic_proposal_procedure_data(protocol),
            "family": family.to_data(),
            "policy": policy.to_data(),
        },
        sort_keys=True,
    ).lower()
    # Match identities, not the final letters of the ordinary word "boolean".
    for forbidden in ('"lean"', '"backend"', '"checker"', "proof_assistant"):
        assert forbidden not in serialized

from __future__ import annotations

from dataclasses import replace
import json

import pytest

from bongard.direct_visual_leg import DirectVisualLowering
from bongard.ir import AllOf, Atom
from bongard.semantic_protocol import (
    build_prospective_soft_scorer_protocol,
    build_visual_semantic_policy,
    visual_semantic_proposal_procedure_digest,
)
from bongard.semantic_synthesis import (
    DIRECT_BOUNDARY_NAME,
    SOFT_BOUNDARY_NAME,
    SemanticSynthesisError,
    VisualSemanticLoweringArchive,
    compile_visual_semantic_proposal,
    visual_proposal_protocol_digest,
)
from bongard.soft_predicates import SoftFamilyDevelopmentUnit, SoftScorerFamily
from bongard.typed_visual_proposal import (
    PANEL_DESCRIPTION_KEYS,
    TypedVisualProposal,
    parse_typed_visual_proposal,
)
from bongard.visual_predicate_catalog import DIRECT_VISUAL_ATOM_CATALOG
from bongard.visual_witness_bundle import VISUAL_WITNESS_BUNDLE
from bongard.legs import FROZEN_VISUAL_SCORE


def _digest(label: str) -> str:
    import hashlib

    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _family() -> SoftScorerFamily:
    protocol = build_prospective_soft_scorer_protocol(
        proposer_model_id="fixture-proposer",
        proposer_reasoning_effort="medium",
        scorer_model_id="fixture-scorer",
        scorer_reasoning_effort="medium",
        score_bin_edges=(0.0, 0.5, 1.0),
        affirmative_boundary=0.5,
        confidence_level=0.1,
        minimum_clusters_per_bin=8,
    )
    units = []
    for score, label, prefix, bin_index in (
        (0.0, False, "low", 0),
        (1.0, True, "high", 1),
    ):
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
                    affirmative_label=label,
                    score_bin_index=bin_index,
                )
            )
    return SoftScorerFamily.fit(
        protocol,
        tuple(sorted(units, key=lambda item: item.observation_id)),
        expected_protocol_digest=protocol.digest(),
    )


def _proposal(family: SoftScorerFamily, kind: str) -> TypedVisualProposal:
    if kind not in {"direct", "soft", "mixed"}:
        raise ValueError(kind)
    direct = (
        []
        if kind == "soft"
        else [
            {
                "catalog_key": "component.count",
                "comparison": "equal",
                "arguments": {"target_count": 2},
            },
            {
                "catalog_key": "hole.owner_count",
                "comparison": "equal",
                "arguments": {"target_count": 1},
            },
        ]
    )
    soft = (
        None
        if kind == "direct"
        else {
            "positive_description": "bird-like articulated organization",
            "cue_descriptions": [
                "one compact central body mass",
                "two lateral wing-like extensions",
            ],
        }
    )
    count = len(direct) + (soft is not None)
    return parse_typed_visual_proposal(
        {
            "positive_description": "a compact articulated ink arrangement",
            "panel_descriptions": {
                name: f"literal drawing {index}"
                for index, name in enumerate(PANEL_DESCRIPTION_KEYS)
            },
            "view": "carrier_shape",
            "deterministic_atoms": direct,
            "soft_claim": soft,
            "formula": {"kind": "all", "atom_indices": list(range(count))},
        },
        catalog=DIRECT_VISUAL_ATOM_CATALOG,
        scorer_protocol_digest=family.protocol_digest,
    )


@pytest.mark.parametrize(
    ("kind", "formula_type", "boundaries", "mapping_ids"),
    [
        (
            "direct",
            Atom,
            {DIRECT_BOUNDARY_NAME},
            ["direct-composite"],
        ),
        ("soft", Atom, {SOFT_BOUNDARY_NAME}, ["soft-calibrated"]),
        (
            "mixed",
            AllOf,
            {DIRECT_BOUNDARY_NAME, SOFT_BOUNDARY_NAME},
            ["direct-composite", "soft-calibrated"],
        ),
    ],
)
def test_compile_direct_soft_and_mixed_shapes(
    kind, formula_type, boundaries, mapping_ids
) -> None:
    family = _family()
    policy = build_visual_semantic_policy(
        family, prospective_protocol=family.protocol
    )
    result = compile_visual_semantic_proposal(
        _proposal(family, kind),
        policy=policy,
        expected_policy_digest=policy.digest(),
        family=family,
    )

    assert isinstance(result.formula, formula_type)
    assert result.registry.frozen
    assert set(result.boundary_types) == boundaries
    assert result.boundary_types.get(DIRECT_BOUNDARY_NAME) in {
        None,
        VISUAL_WITNESS_BUNDLE,
    }
    assert result.boundary_types.get(SOFT_BOUNDARY_NAME) in {
        None,
        FROZEN_VISUAL_SCORE,
    }
    assert [
        item.composite_id for item in result.lowering_archive.composite_mapping
    ] == mapping_ids
    result.attachment_contract.validate(result.formula, result.registry)


def test_direct_atoms_lower_to_one_composite_and_archive_round_trips() -> None:
    family = _family()
    policy = build_visual_semantic_policy(
        family, prospective_protocol=family.protocol
    )
    proposal = _proposal(family, "mixed")
    result = compile_visual_semantic_proposal(
        proposal,
        policy=policy,
        expected_policy_digest=policy.digest(),
        family=family,
    )
    archive = result.lowering_archive

    assert isinstance(archive.direct_lowering, DirectVisualLowering)
    assert archive.direct_lowering.atom_ids == ("atom-00", "atom-01")
    assert archive.soft_lowering is not None
    assert archive.soft_lowering.claim.atom_id == "atom-02"
    assert archive.original_formula_atom_ids == (
        "atom-00",
        "atom-01",
        "atom-02",
    )
    assert archive.composite_mapping[0].source_atom_ids == (
        "atom-00",
        "atom-01",
    )
    assert archive.composite_mapping[1].source_atom_ids == ("atom-02",)
    assert VisualSemanticLoweringArchive.from_data(archive.to_data()) == archive
    assert len(archive.digest) == 64

    encoded = json.dumps(archive.to_data(), sort_keys=True).lower()
    assert "lean" not in encoded
    assert "backend" not in encoded
    assert archive.proposal_protocol_digest == (
        visual_semantic_proposal_procedure_digest(family.protocol)
    )
    assert visual_proposal_protocol_digest(family.protocol) == (
        archive.proposal_protocol_digest
    )


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("direct_predicate_catalog_digest", "direct predicate catalog"),
        ("proposal_protocol_digest", "proposal protocol"),
        ("soft_scorer_protocol_digest", "scorer protocol"),
        ("soft_scorer_family_digest", "scorer family"),
        (
            "soft_family_development_manifest_digest",
            "family development manifest",
        ),
        ("witness_catalog_digest", "witness catalog"),
    ],
)
def test_compiler_rejects_every_policy_dependency_mismatch(field, message) -> None:
    family = _family()
    policy = build_visual_semantic_policy(
        family, prospective_protocol=family.protocol
    )
    changed = replace(policy, **{field: _digest("changed-" + field)})
    with pytest.raises(SemanticSynthesisError, match=message):
        compile_visual_semantic_proposal(
            _proposal(family, "mixed"),
            policy=changed,
            expected_policy_digest=changed.digest(),
            family=family,
        )


def test_compiler_rejects_wrong_policy_commitment_and_unsupported_proposal() -> None:
    family = _family()
    policy = build_visual_semantic_policy(
        family, prospective_protocol=family.protocol
    )
    proposal = _proposal(family, "direct")
    with pytest.raises(SemanticSynthesisError, match="differs from commitment"):
        compile_visual_semantic_proposal(
            proposal,
            policy=policy,
            expected_policy_digest="0" * 64,
            family=family,
        )

    unsupported = replace(proposal, catalog_digest="0" * 64)
    with pytest.raises(SemanticSynthesisError, match="different direct catalog"):
        compile_visual_semantic_proposal(
            unsupported,
            policy=policy,
            expected_policy_digest=policy.digest(),
            family=family,
        )


def test_compiler_rejects_forged_empty_typed_proposal() -> None:
    family = _family()
    policy = build_visual_semantic_policy(
        family, prospective_protocol=family.protocol
    )
    proposal = _proposal(family, "direct")
    object.__setattr__(proposal, "deterministic_atoms", ())

    with pytest.raises(SemanticSynthesisError):
        compile_visual_semantic_proposal(
            proposal,
            policy=policy,
            expected_policy_digest=policy.digest(),
            family=family,
        )

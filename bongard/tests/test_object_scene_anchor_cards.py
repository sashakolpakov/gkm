from __future__ import annotations

from copy import deepcopy
import hashlib

import numpy as np
import pytest

from bongard.canonical import canonical_digest
from bongard.object_scene_anchor_bindings import ObjectSceneAnchorBindingSpec
from bongard.object_scene_anchor_cards import (
    OBJECT_SCENE_ANCHOR_CARD_WITNESS_KINDS,
    OBJECT_SCENE_ANCHOR_CARD_WITNESS_SCHEMA,
    ObjectSceneAnchorCardError,
    ObjectSceneAnchorCardProposal,
    ObjectSceneAnchorCardWitness,
    build_object_scene_anchor_card_proposal,
    verify_object_scene_anchor_card_proposal,
)
from bongard.object_scene_anchor_catalog import ObjectSceneAnchorDecisionManifest
from bongard.object_scene_anchor_panel_manifest import (
    ObjectSceneAnchorPanelDecisionManifest,
    _manifest_content,
)
from bongard.object_scene_anchor_salience import extract_object_scene_anchor_salience


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def _plus() -> np.ndarray:
    mask = np.zeros((43, 43), dtype=np.bool_)
    mask[21, 7:36] = True
    mask[7:36, 21] = True
    return mask


def _panel_manifest(
    index: int,
    decisions: tuple[ObjectSceneAnchorDecisionManifest, ...],
) -> ObjectSceneAnchorPanelDecisionManifest:
    values = {
        "panel_digest": _sha(f"panel-{index}"),
        "width_pixels": 64,
        "height_pixels": 64,
        "inventory_digest": _sha(f"inventory-{index}"),
        "proposal_count": len(decisions),
        "object_ids": tuple(item.object_id for item in decisions),
        "object_decisions": decisions,
    }
    provisional = object.__new__(ObjectSceneAnchorPanelDecisionManifest)
    for name, item in values.items():
        object.__setattr__(provisional, name, item)
    return ObjectSceneAnchorPanelDecisionManifest(
        **values,
        manifest_digest=canonical_digest(_manifest_content(provisional)),
    )


@pytest.fixture(scope="module")
def panel_sets():
    decisions = tuple(
        ObjectSceneAnchorDecisionManifest.from_salience(
            extract_object_scene_anchor_salience(_plus(), f"object_{index:04d}")
        )
        for index in range(2)
    )
    manifests = {
        f"panel_{index:03d}": _panel_manifest(index, decisions)
        for index in range(12)
    }
    return (
        {key: manifests[key] for key in tuple(manifests)[:6]},
        {key: manifests[key] for key in tuple(manifests)[6:]},
    )


def _citations(aliases, *, object_id: str = "object_0000", anchor_id: str = "entity"):
    return [
        {
            "panel_alias": alias,
            "object_id": object_id,
            "anchor_id": anchor_id,
        }
        for alias in sorted(aliases)
    ]


def _card(aliases, phrase: str, statement: str):
    return {
        "phrase": phrase,
        "binding_spec": ObjectSceneAnchorBindingSpec.entity().to_data(),
        "required_witnesses": [
            {"kind": "shape_appearance", "statement": statement},
            {
                "kind": "part_topology",
                "statement": "the bound form contains one continuous outer path",
            },
        ],
        "accepted_variants": [
            "gentle curvature remains included",
            "softly tapered curvature remains included",
        ],
        "near_miss_boundaries": [
            "a sharply pointed upper tip falls outside",
        ],
        "positive_support_citations": _citations(aliases),
    }


def _payload(side0, side1):
    return {
        "side0_positive": [
            _card(side0, "rounded upper contour", "the bound form has a rounded upper contour"),
            _card(side0, "centered circular marking", "the bound form carries a centered circular mark"),
        ],
        "side1_positive": [
            _card(side1, "angular upper contour", "the bound form has an angular upper contour"),
            _card(side1, "paired internal marking", "the bound form carries paired internal marks"),
        ],
    }


def _keys(value: object):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _keys(item)


def test_roundtrip_exact_citations_and_witness_projection(panel_sets) -> None:
    side0, side1 = panel_sets
    proposal = build_object_scene_anchor_card_proposal(
        _payload(side0, side1),
        side0_panel_manifests=side0,
        side1_panel_manifests=side1,
    )

    assert ObjectSceneAnchorCardProposal.from_data(proposal.to_data()) == proposal
    assert (
        verify_object_scene_anchor_card_proposal(
            proposal,
            side0_panel_manifests=side0,
            side1_panel_manifests=side1,
        )
        == proposal
    )
    cards = (*proposal.side0_positive, *proposal.side1_positive)
    assert tuple(item.card_id for item in cards) == tuple(
        f"card_{index:04d}" for index in range(4)
    )
    assert all(len(item.positive_support_citations) == 6 for item in cards)
    assert all(
        citation.binding_alias == citation.resolved_binding.binding_alias
        and citation.binding_digest == citation.resolved_binding.binding_digest
        and citation.object_id == citation.resolved_binding.object_id
        and citation.resolved_binding.spec_digest == card.binding_spec.spec_digest
        for card in cards
        for citation in card.positive_support_citations
    )
    assert all(
        tuple(spec.witness_digest for spec in card.binding_witness_specs)
        == tuple(item.witness_digest for item in card.required_witnesses)
        for card in cards
    )
    keys = tuple(_keys(proposal.to_data()))
    assert not any("lean" in key.casefold() for key in keys)
    assert not any("raw_graph" in key or "audit_graph" in key for key in keys)
    assert proposal.to_data()["truth_assignment_present"] is False


def test_canonicalization_is_input_order_independent(panel_sets) -> None:
    side0, side1 = panel_sets
    first_payload = _payload(side0, side1)
    second_payload = deepcopy(first_payload)
    second_payload["side0_positive"].reverse()
    second_payload["side1_positive"].reverse()
    for bucket in second_payload.values():
        for card in bucket:
            card["required_witnesses"].reverse()
            card["accepted_variants"].reverse()

    first = build_object_scene_anchor_card_proposal(
        first_payload,
        side0_panel_manifests=side0,
        side1_panel_manifests=side1,
    )
    second = build_object_scene_anchor_card_proposal(
        second_payload,
        side0_panel_manifests=side0,
        side1_panel_manifests=side1,
    )
    assert first == second
    assert OBJECT_SCENE_ANCHOR_CARD_WITNESS_KINDS == (
        "shape_appearance",
        "marking_pattern",
        "spatial_relation",
        "part_topology",
    )

    first_witness = ObjectSceneAnchorCardWitness.create(
        "witness_00", "shape_appearance", "the bound form has a rounded upper contour"
    )
    second_witness = ObjectSceneAnchorCardWitness.create(
        "witness_03", "shape_appearance", "the bound form has a rounded upper contour"
    )
    assert first_witness.witness_digest == second_witness.witness_digest
    assert first_witness.witness_digest == canonical_digest(
        {
            "schema": OBJECT_SCENE_ANCHOR_CARD_WITNESS_SCHEMA,
            "kind": first_witness.kind,
            "statement": first_witness.statement,
        }
    )


@pytest.mark.parametrize(
    "statement",
    (
        "the bound path has at least two visually distinct pronounced bends",
        "the bound form contains at least one loop",
        "the bound object carries at least three visible marks",
        "the bound figure shows at least four separate arms",
        "the bound contour makes at least two sharp turns",
        "the bound shape includes at least two straight strokes",
        "the bound outline has at least three angular corners",
    ),
)
def test_explicit_bound_object_local_feature_lower_bounds_are_allowed(
    statement,
) -> None:
    witness = ObjectSceneAnchorCardWitness.create(
        "witness_00", "part_topology", statement
    )
    assert witness.statement == statement


@pytest.mark.parametrize(
    "statement",
    (
        "the bound form contains at least two visible objects",
        "the bound object contains at least two distinct entities",
        "the bound figure contains at least two separate figures",
        "the bound form is at least two times larger",
        "the bound form has at least two times larger loops",
        "at least two bends define the bound path",
        "the path has at least two bends",
        "the bound path has at least nine bends",
    ),
)
def test_nonlocal_or_comparative_lower_bounds_are_rejected(statement) -> None:
    with pytest.raises(ObjectSceneAnchorCardError, match="anchor-local prose"):
        ObjectSceneAnchorCardWitness.create(
            "witness_00",
            "part_topology",
            statement,
        )


def test_superlative_and_cross_object_comparisons_remain_rejected() -> None:
    with pytest.raises(ObjectSceneAnchorCardError, match="anchor-local prose"):
        ObjectSceneAnchorCardWitness.create(
            "witness_00",
            "shape_appearance",
            "the bound form has the least curved outline",
        )
    with pytest.raises(ObjectSceneAnchorCardError, match="anchor-local prose"):
        ObjectSceneAnchorCardWitness.create(
            "witness_00",
            "part_topology",
            "the bound path has at least two bends across objects",
        )


def test_malformed_optional_card_is_dropped_without_payload_identity(panel_sets) -> None:
    side0, side1 = panel_sets
    payload = _payload(side0, side1)
    payload["side0_positive"][1] = {"phrase": "broken row"}
    proposal = build_object_scene_anchor_card_proposal(
        payload,
        side0_panel_manifests=side0,
        side1_panel_manifests=side1,
    )

    assert len(proposal.side0_positive) == 1
    assert len(proposal.side1_positive) == 2
    assert len(proposal.dropped_cards) == 1
    assert proposal.dropped_cards[0].reason_code == "malformed_card"
    dropped = proposal.dropped_cards[0].to_data()
    assert "payload_digest" not in dropped
    assert "payload" not in dropped
    assert "phrase" not in dropped


@pytest.mark.parametrize(
    ("field", "bad_value", "reason"),
    (
        ("phrase", "not rounded upper contour", "phrase_policy"),
        (
            "required_witnesses",
            [
                {
                    "kind": "shape_appearance",
                    "statement": "the bound form is not rounded along its upper contour",
                }
            ],
            "witness_policy",
        ),
        (
            "accepted_variants",
            ["a version without rounded curvature remains included"],
            "variant_policy",
        ),
        (
            "required_witnesses",
            [
                {
                    "kind": "count_relation",
                    "statement": "the bound form contains three visible turns",
                }
            ],
            "witness_policy",
        ),
    ),
)
def test_negation_or_foreign_witness_kind_never_becomes_an_atom(
    panel_sets, field, bad_value, reason
) -> None:
    side0, side1 = panel_sets
    payload = _payload(side0, side1)
    payload["side0_positive"][0][field] = bad_value
    proposal = build_object_scene_anchor_card_proposal(
        payload,
        side0_panel_manifests=side0,
        side1_panel_manifests=side1,
    )
    assert any(item.reason_code == reason for item in proposal.dropped_cards)
    assert all(
        " not " not in f" {witness.statement} "
        for card in (*proposal.side0_positive, *proposal.side1_positive)
        for witness in card.required_witnesses
    )


def test_foreign_object_or_anchor_mismatch_is_dropped(panel_sets) -> None:
    side0, side1 = panel_sets
    payload = _payload(side0, side1)
    payload["side0_positive"][0]["positive_support_citations"][0][
        "object_id"
    ] = "object_9999"
    payload["side1_positive"][0]["positive_support_citations"][0][
        "anchor_id"
    ] = "part-99999999"
    proposal = build_object_scene_anchor_card_proposal(
        payload,
        side0_panel_manifests=side0,
        side1_panel_manifests=side1,
    )
    assert {item.reason_code for item in proposal.dropped_cards} == {
        "foreign_object",
        "binding_mismatch",
    }
    assert len(proposal.side0_positive) == len(proposal.side1_positive) == 1


def test_strict_from_data_rejects_resealed_binding_spec_mismatch(panel_sets) -> None:
    side0, side1 = panel_sets
    proposal = build_object_scene_anchor_card_proposal(
        _payload(side0, side1),
        side0_panel_manifests=side0,
        side1_panel_manifests=side1,
    )
    data = deepcopy(proposal.to_data())
    card = data["side0_positive"][0]
    card["binding_spec"] = ObjectSceneAnchorBindingSpec.part().to_data()
    card["card_digest"] = canonical_digest(
        {key: item for key, item in card.items() if key != "card_digest"}
    )
    data["proposal_digest"] = canonical_digest(
        {key: item for key, item in data.items() if key != "proposal_digest"}
    )
    with pytest.raises(ObjectSceneAnchorCardError, match="shared binding spec"):
        ObjectSceneAnchorCardProposal.from_data(data)

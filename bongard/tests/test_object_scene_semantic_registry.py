from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import inspect
from pathlib import Path

import pytest

from bongard.object_scene_semantic_registry import (
    MAX_CONCEPTS_PER_ORIENTATION,
    ObjectScenePreparedSemanticRegistryProposal,
    ObjectSceneSemanticRegistryError,
    ObjectSceneSemanticRegistryProposal,
    build_object_scene_semantic_registry_gap,
    build_object_scene_semantic_registry_proposal,
    object_scene_semantic_registry_protocol_digest,
    object_scene_semantic_registry_source_digest,
    prepare_object_scene_semantic_registry_proposal,
    verify_object_scene_semantic_registry_proposal,
)
from bongard.object_scene_visual_frontend import (
    ObjectSceneSoftTagRegistry,
    ObjectSceneTranscriptMode,
    extract_object_scene_proposal_inventory,
    observe_object_scene_transcript,
)
from bongard.tests.test_object_scene_visual_frontend import (
    _payload,
    _scene,
    _transport,
)
from bongard.tests.test_prototype_scene_observer import (
    EFFORT,
    LAUNCHER_DIGEST,
    MODEL,
    MODEL_CATALOG,
    NO_TOOLS_ATTESTATION,
)


def _discovery_artifact(index: int):
    raw = _scene(index)
    inventory = extract_object_scene_proposal_inventory(raw)
    return observe_object_scene_transcript(
        raw,
        scene_id=f"secret-task-panel-{index:02d}",
        observation_context_digest=(
            "sha256:"
            + hashlib.sha256(f"context-{index}".encode("ascii")).hexdigest()
        ),
        mode=ObjectSceneTranscriptMode.DISCOVERY,
        registry=None,
        model=MODEL,
        reasoning_effort=EFFORT,
        expected_launcher_digest=LAUNCHER_DIGEST,
        model_catalog_snapshot=MODEL_CATALOG,
        no_tools_attestation=NO_TOOLS_ATTESTATION,
        transport=_transport(
            _payload(
                inventory,
                open_tags=("bird-like object", "mismatched parts"),
                panel_open_tags=("paired visible forms",),
            ),
            [],
        ),
    )


@pytest.fixture(scope="module")
def discovery_inputs():
    artifacts = tuple(_discovery_artifact(index) for index in range(12))
    roles = tuple(
        {
            "ordinal": index,
            "neutral_panel_digest": hashlib.sha256(
                f"neutral-{index}".encode("ascii")
            ).hexdigest(),
            "historical_role": index // 6,
            "blind_panel_id": artifact.scene_id,
        }
        for index, artifact in enumerate(artifacts)
    )
    return artifacts, roles


def _aliases(prepared, side: int) -> tuple[str, ...]:
    return tuple(
        item["alias"]
        for item in prepared.alias_bindings
        if item["historical_role"] == side and item["usable"] is True
    )


def _concept(scope: str, phrase: str, citations) -> dict[str, object]:
    return {"scope": scope, "phrase": phrase, "citations": list(citations)}


def _valid_payload(prepared):
    side0 = _aliases(prepared, 0)
    side1 = _aliases(prepared, 1)
    return {
        "side0_positive": [
            _concept("panel", "paired visible forms", side0[:3]),
            _concept("entity", "mismatched parts", side0[:2]),
        ],
        "side1_positive": [
            _concept("entity", "unequal edge lengths", side1[:4]),
            _concept("panel", "balanced spacing", side1[:2]),
        ],
    }


def test_prepare_is_order_invariant_opaque_text_only_and_strict(discovery_inputs):
    artifacts, roles = discovery_inputs
    prepared = prepare_object_scene_semantic_registry_proposal(artifacts, roles)
    reversed_prepared = prepare_object_scene_semantic_registry_proposal(
        tuple(reversed(artifacts)), tuple(reversed(roles))
    )

    assert reversed_prepared == prepared
    assert ObjectScenePreparedSemanticRegistryProposal.from_data(
        prepared.to_data()
    ) == prepared
    assert len(prepared.alias_bindings) == 12
    assert tuple(item["alias"] for item in prepared.alias_bindings) == tuple(
        f"panel_{index:03d}" for index in range(12)
    )
    assert "outlined visible form" in prepared.prompt
    assert "bird-like object" in prepared.prompt
    assert prepared.output_schema["additionalProperties"] is False
    assert set(prepared.output_schema["required"]) == {
        "side0_positive", "side1_positive"
    }

    lowered = prepared.prompt.lower()
    for forbidden in (
        "pixel", "image", "formula", "candidate_digest", "object_0000",
        "secret-task-panel", "neutral_panel_digest", "artifact_digest",
        "panel_digest", "transcript_digest",
    ):
        assert forbidden not in lowered
    for artifact in artifacts:
        assert artifact.scene_id not in prepared.prompt
        assert artifact.artifact_digest not in prepared.prompt
        assert artifact.panel_digest not in prepared.prompt


def test_digest_shuffle_is_not_input_order_and_role_swap_swaps_buckets(
    discovery_inputs,
):
    artifacts, roles = discovery_inputs
    prepared = prepare_object_scene_semantic_registry_proposal(artifacts, roles)
    assert tuple(item["scene_id"] for item in prepared.alias_bindings) != tuple(
        artifact.scene_id for artifact in artifacts
    )

    swapped_roles = tuple(
        {**row, "historical_role": 1 - row["historical_role"]} for row in roles
    )
    swapped = prepare_object_scene_semantic_registry_proposal(
        artifacts, swapped_roles
    )
    assert swapped.model_view["side0_support_descriptions"] == prepared.model_view[
        "side1_support_descriptions"
    ]
    assert swapped.model_view["side1_support_descriptions"] == prepared.model_view[
        "side0_support_descriptions"
    ]
    assert _aliases(swapped, 0) == _aliases(prepared, 1)
    assert _aliases(swapped, 1) == _aliases(prepared, 0)


def test_valid_both_bucket_union_uses_citations_for_counts_and_order(
    discovery_inputs,
):
    artifacts, roles = discovery_inputs
    prepared = prepare_object_scene_semantic_registry_proposal(artifacts, roles)
    payload = _valid_payload(prepared)
    proposal, registry = build_object_scene_semantic_registry_proposal(
        prepared, payload
    )

    assert proposal.status == "proposed"
    assert tuple(
        (item.tag_id, item.scope, item.tag, item.distinct_panel_count)
        for item in registry.tags
    ) == (
        ("tag_0000", "entity", "unequal edge lengths", 4),
        ("tag_0001", "panel", "paired visible forms", 3),
        ("tag_0002", "entity", "mismatched parts", 2),
        ("tag_0003", "panel", "balanced spacing", 2),
    )
    assert {item.scope for item in registry.tags} == {"panel", "entity"}
    assert registry.source_panel_digests == tuple(
        sorted(artifact.panel_digest for artifact in artifacts)
    )


def test_same_phrase_may_exist_at_panel_and_entity_scope(discovery_inputs):
    artifacts, roles = discovery_inputs
    prepared = prepare_object_scene_semantic_registry_proposal(artifacts, roles)
    payload = {
        "side0_positive": [
            _concept("panel", "pointed form", _aliases(prepared, 0)[:2])
        ],
        "side1_positive": [
            _concept("entity", "pointed form", _aliases(prepared, 1)[:2])
        ],
    }
    _, registry = build_object_scene_semantic_registry_proposal(prepared, payload)
    assert tuple((item.scope, item.tag) for item in registry.tags) == (
        ("entity", "pointed form"), ("panel", "pointed form")
    )


@pytest.mark.parametrize(
    "phrase",
    (
        "not pointed",
        "without internal marks",
        "pointed and curved",
        "pointed or curved",
        "more common forms",
        "side marker",
        "usually pointed",
        "never contains circles",
        "isn't circular",
        "noncircular form",
        "avoids circles",
        "pointed plus curved",
        "pointed also curved",
        "higher occurrence",
        "rarer pattern",
        "dominant pattern",
        "exclusive circles",
        "contrastive curvedness",
    ),
)
def test_rejects_negation_logical_packaging_and_role_comparison(
    discovery_inputs, phrase,
):
    artifacts, roles = discovery_inputs
    prepared = prepare_object_scene_semantic_registry_proposal(artifacts, roles)
    payload = _valid_payload(prepared)
    payload["side0_positive"][0]["phrase"] = phrase
    with pytest.raises(ObjectSceneSemanticRegistryError):
        build_object_scene_semantic_registry_proposal(prepared, payload)


def test_rejects_duplicate_scoped_phrase_and_duplicate_citation(discovery_inputs):
    artifacts, roles = discovery_inputs
    prepared = prepare_object_scene_semantic_registry_proposal(artifacts, roles)
    payload = _valid_payload(prepared)
    payload["side1_positive"][1]["scope"] = "entity"
    payload["side1_positive"][1]["phrase"] = "mismatched parts"
    with pytest.raises(ObjectSceneSemanticRegistryError, match="repeats"):
        build_object_scene_semantic_registry_proposal(prepared, payload)

    payload = _valid_payload(prepared)
    alias = _aliases(prepared, 0)[0]
    payload["side0_positive"][0]["citations"] = [alias, alias]
    with pytest.raises(ObjectSceneSemanticRegistryError, match="distinct"):
        build_object_scene_semantic_registry_proposal(prepared, payload)


def test_rejects_foreign_and_cross_side_citations(discovery_inputs):
    artifacts, roles = discovery_inputs
    prepared = prepare_object_scene_semantic_registry_proposal(artifacts, roles)
    for citation in ("panel_999", _aliases(prepared, 1)[0]):
        payload = _valid_payload(prepared)
        payload["side0_positive"][0]["citations"] = [
            _aliases(prepared, 0)[0], citation
        ]
        with pytest.raises(ObjectSceneSemanticRegistryError, match="foreign|cross-side"):
            build_object_scene_semantic_registry_proposal(prepared, payload)


def test_rejects_malformed_citation_items_with_typed_payload_error(
    discovery_inputs,
):
    artifacts, roles = discovery_inputs
    prepared = prepare_object_scene_semantic_registry_proposal(artifacts, roles)
    for malformed in ([{}, {}], [1, 2], [_aliases(prepared, 0)[0], 2]):
        payload = _valid_payload(prepared)
        payload["side0_positive"][0]["citations"] = malformed
        with pytest.raises(ObjectSceneSemanticRegistryError):
            build_object_scene_semantic_registry_proposal(prepared, payload)


def test_prepare_requires_exact_twelve_and_six_per_role(discovery_inputs):
    artifacts, roles = discovery_inputs
    with pytest.raises(ObjectSceneSemanticRegistryError, match="exactly twelve"):
        prepare_object_scene_semantic_registry_proposal(artifacts[:-1], roles[:-1])
    imbalanced = tuple(
        {**row, "historical_role": 0 if index < 7 else 1}
        for index, row in enumerate(roles)
    )
    with pytest.raises(ObjectSceneSemanticRegistryError, match="exactly six"):
        prepare_object_scene_semantic_registry_proposal(artifacts, imbalanced)


def test_rejects_empty_and_capacity_excess(discovery_inputs):
    artifacts, roles = discovery_inputs
    prepared = prepare_object_scene_semantic_registry_proposal(artifacts, roles)
    side0 = _aliases(prepared, 0)
    side1 = _aliases(prepared, 1)
    with pytest.raises(ObjectSceneSemanticRegistryError, match="capacity"):
        build_object_scene_semantic_registry_proposal(
            prepared,
            {
                "side0_positive": [],
                "side1_positive": [_concept("panel", "balanced spacing", side1[:2])],
            },
        )

    seventeen = [
        _concept("panel", f"shape phrase {chr(97 + index)}", side0[:2])
        for index in range(MAX_CONCEPTS_PER_ORIENTATION + 1)
    ]
    with pytest.raises(ObjectSceneSemanticRegistryError, match="capacity"):
        build_object_scene_semantic_registry_proposal(
            prepared,
            {
                "side0_positive": seventeen,
                "side1_positive": [_concept("entity", "mismatched parts", side1[:2])],
            },
        )

    with pytest.raises(ObjectSceneSemanticRegistryError, match="capacity"):
        build_object_scene_semantic_registry_proposal(
            prepared,
            {
                "side0_positive": seventeen,
                "side1_positive": seventeen,
            },
        )


def test_roundtrip_and_companion_verifier_reconstruct_without_frequency_freezer(
    discovery_inputs,
):
    artifacts, roles = discovery_inputs
    prepared = prepare_object_scene_semantic_registry_proposal(artifacts, roles)
    proposal, registry = build_object_scene_semantic_registry_proposal(
        prepared, _valid_payload(prepared)
    )
    restored = ObjectSceneSemanticRegistryProposal.from_data(proposal.to_data())
    restored_registry = ObjectSceneSoftTagRegistry.from_data(registry.to_data())
    assert restored == proposal
    assert restored_registry == registry
    assert verify_object_scene_semantic_registry_proposal(
        proposal.to_data(), restored_registry, artifacts, roles
    ) == proposal
    verifier_source = inspect.getsource(
        verify_object_scene_semantic_registry_proposal
    )
    assert "verify_object_scene_soft_tag_registry" not in verifier_source
    assert "freeze_object_scene_soft_tag_registry" not in verifier_source


def test_proposal_and_prepared_tampering_is_rejected(discovery_inputs):
    artifacts, roles = discovery_inputs
    prepared = prepare_object_scene_semantic_registry_proposal(artifacts, roles)
    proposal, registry = build_object_scene_semantic_registry_proposal(
        prepared, _valid_payload(prepared)
    )

    for field, replacement in (
        ("source_digest", "0" * 64),
        ("protocol_digest", "1" * 64),
        ("role_rows_digest", "2" * 64),
        ("registry_digest", "3" * 64),
    ):
        raw = proposal.to_data()
        raw[field] = replacement
        with pytest.raises(ObjectSceneSemanticRegistryError):
            verify_object_scene_semantic_registry_proposal(raw, registry, artifacts, roles)

    raw = proposal.to_data()
    raw["model_payload"]["side0_positive"][0]["phrase"] = "pointed form"
    with pytest.raises(ObjectSceneSemanticRegistryError):
        verify_object_scene_semantic_registry_proposal(raw, registry, artifacts, roles)

    prepared_raw = prepared.to_data()
    prepared_raw["alias_bindings"][0]["historical_role"] = (
        1 - prepared_raw["alias_bindings"][0]["historical_role"]
    )
    with pytest.raises(ObjectSceneSemanticRegistryError):
        ObjectScenePreparedSemanticRegistryProposal.from_data(prepared_raw)

    altered_roles = tuple(
        ({**row, "historical_role": 1 - row["historical_role"]} if index == 0 else row)
        for index, row in enumerate(roles)
    )
    with pytest.raises(ObjectSceneSemanticRegistryError):
        verify_object_scene_semantic_registry_proposal(
            proposal, registry, artifacts, altered_roles
        )

    registry_raw = registry.to_data()
    registry_raw["registry_digest"] = "4" * 64
    with pytest.raises(Exception):
        verify_object_scene_semantic_registry_proposal(
            proposal, ObjectSceneSoftTagRegistry.from_data(registry_raw), artifacts, roles
        )


@pytest.mark.parametrize(
    "gap_code", ("payload_rejected", "insufficient_discovery_evidence")
)
def test_typed_gap_has_zero_tags_and_verifies(discovery_inputs, gap_code):
    artifacts, roles = discovery_inputs
    prepared = prepare_object_scene_semantic_registry_proposal(artifacts, roles)
    rejected_payload = (
        {"side0_positive": [], "side1_positive": []}
        if gap_code == "payload_rejected"
        else None
    )
    proposal, registry = build_object_scene_semantic_registry_gap(
        prepared, gap_code, rejected_payload
    )
    assert proposal.status == "typed_proposal_gap"
    assert proposal.model_payload == rejected_payload
    assert proposal.side0_positive == proposal.side1_positive == ()
    assert registry.tags == ()
    assert verify_object_scene_semantic_registry_proposal(
        proposal, registry, artifacts, roles
    ) == proposal


def test_source_protocol_digest_and_python_only_surface_are_stable():
    source_path = Path(inspect.getsourcefile(prepare_object_scene_semantic_registry_proposal))
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imports = tuple(
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    ) + tuple(
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    )
    assert all("lean" not in item.lower() for item in imports)
    assert len(object_scene_semantic_registry_source_digest()) == 64
    assert len(object_scene_semantic_registry_protocol_digest()) == 64

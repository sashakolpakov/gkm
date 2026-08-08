from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import inspect
from pathlib import Path

import pytest

import bongard.object_scene_semantic_registry as semantic_registry
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


def _concept(
    scope: str,
    phrase: str,
    citations,
    *,
    required_witnesses=None,
    accepted_variants=(),
    near_miss_boundaries=(),
) -> dict[str, object]:
    witnesses = (
        [{"kind": "shape_appearance", "statement": phrase}]
        if required_witnesses is None
        else list(required_witnesses)
    )
    return {
        "scope": scope,
        "phrase": phrase,
        "required_witnesses": witnesses,
        "accepted_variants": list(accepted_variants),
        "near_miss_boundaries": list(near_miss_boundaries),
        "citations": list(citations),
    }


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
    concept_schema = prepared.output_schema["properties"]["side0_positive"][
        "items"
    ]
    assert set(concept_schema["required"]) == {
        "scope",
        "phrase",
        "required_witnesses",
        "accepted_variants",
        "near_miss_boundaries",
        "citations",
    }
    assert "near-miss boundaries are the sole exception" in prepared.prompt.lower()
    assert "using 'does not qualify', 'is excluded', or 'falls outside'" in (
        prepared.prompt
    )
    assert "never join alternative cues with 'or' or 'either'" in prepared.prompt
    assert "these may enumerate alternatives with canonical commas" in (
        prepared.prompt
    )
    assert "visible paths, and visible sides are affirmative and allowed" in (
        prepared.prompt
    )
    assert "the configuration before that exclusion must itself be affirmative" in (
        prepared.prompt.lower()
    )
    witness_description = concept_schema["properties"]["required_witnesses"][
        "description"
    ]
    variant_description = concept_schema["properties"]["accepted_variants"][
        "description"
    ]
    boundary_description = concept_schema["properties"]["near_miss_boundaries"][
        "description"
    ]
    assert "one atomic check" in witness_description
    assert "comma-space lists and or are allowed here" in variant_description
    assert "exactly one controlled exclusion phrase" in boundary_description

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
    assert proposal.dropped_concepts == ()
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
    assert all(len(item.required_witnesses) == 1 for item in registry.tags)
    assert all(item.required_witnesses[0].witness_id == "witness_00" for item in registry.tags)
    assert all(len(item.criteria_digest) == 64 for item in registry.tags)
    assert all(len(item.tag_digest) == 64 for item in registry.tags)
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


def test_transparent_witness_macro_is_canonical_and_digest_bound(
    discovery_inputs,
):
    artifacts, roles = discovery_inputs
    prepared = prepare_object_scene_semantic_registry_proposal(artifacts, roles)
    payload = _valid_payload(prepared)
    payload["side0_positive"][1] = _concept(
        "entity",
        "mismatched upper and lower portions",
        _aliases(prepared, 0)[:2],
        required_witnesses=(
            {
                "kind": "shape_appearance",
                "statement": (
                    "corresponding upper and lower portions visibly differ in outline"
                ),
            },
            {
                "kind": "part_topology",
                "statement": (
                    "two visible paths join along one narrow side"
                ),
            },
        ),
        accepted_variants=(
            "fan-like, sector-like, or wedge-like portions count as tapered portions",
        ),
        near_miss_boundaries=(
            "one closed triangle made from three strands does not qualify",
        ),
    )
    proposal, registry = build_object_scene_semantic_registry_proposal(
        prepared, payload
    )

    concept = next(
        item
        for item in proposal.side0_positive
        if item.phrase == "mismatched upper and lower portions"
    )
    tag = next(item for item in registry.tags if item.tag == concept.phrase)
    assert tuple(item.kind for item in concept.required_witnesses) == (
        "part_topology",
        "shape_appearance",
    )
    assert tuple(item.witness_id for item in concept.required_witnesses) == (
        "witness_00",
        "witness_01",
    )
    assert tag.required_witnesses == concept.required_witnesses
    assert tag.accepted_variants == concept.accepted_variants
    assert tag.accepted_variants == (
        "fan-like, sector-like, or wedge-like portions count as tapered portions",
    )
    assert tag.near_miss_boundaries == concept.near_miss_boundaries
    assert tag.criteria_digest == concept.criteria_digest

    concept_raw = concept.to_data()
    concept_raw["required_witnesses"][0]["statement"] = (
        "three joined portions share one narrow waist"
    )
    with pytest.raises(ObjectSceneSemanticRegistryError):
        type(concept).from_data(concept_raw)

    registry_raw = registry.to_data()
    registry_index = next(
        index
        for index, item in enumerate(registry_raw["tags"])
        if item["tag_id"] == tag.tag_id
    )
    registry_raw["tags"][registry_index]["near_miss_boundaries"] = []
    with pytest.raises(Exception):
        ObjectSceneSoftTagRegistry.from_data(registry_raw)


@pytest.mark.parametrize(
    "field,replacement",
    (
        ("required_witnesses", []),
        (
            "required_witnesses",
            [
                {
                    "kind": "shape_appearance",
                    "statement": f"visible pointed form number {index}",
                }
                for index in range(4)
            ],
        ),
        (
            "accepted_variants",
            [
                "rounded corners count as pointed corners",
                "short corners count as pointed corners",
                "broad corners count as pointed corners",
            ],
        ),
        (
            "near_miss_boundaries",
            [
                "one rounded form does not qualify",
                "one square form does not qualify",
                "one circular form does not qualify",
            ],
        ),
        (
            "required_witnesses",
            [
                {
                    "kind": "shape_appearance",
                    "statement": "this is common across support panels",
                }
            ],
        ),
        (
            "required_witnesses",
            [
                {
                    "kind": "shape_appearance",
                    "statement": "the entity has a pointed tip or curved edge",
                }
            ],
        ),
    ),
)
def test_bad_operational_card_is_quarantined_without_losing_valid_rows(
    discovery_inputs, field, replacement,
):
    artifacts, roles = discovery_inputs
    prepared = prepare_object_scene_semantic_registry_proposal(artifacts, roles)
    payload = _valid_payload(prepared)
    payload["side0_positive"][0][field] = replacement
    proposal, registry = build_object_scene_semantic_registry_proposal(
        prepared, payload
    )
    assert tuple(item.reason_code for item in proposal.dropped_concepts) == (
        "criteria_policy",
    )
    assert "paired visible forms" not in {item.tag for item in registry.tags}
    assert "mismatched parts" in {item.tag for item in registry.tags}


def test_exact_v4_proposer_cards_keep_variant_lists_nonvoting_and_witnesses_atomic(
    discovery_inputs,
):
    artifacts, roles = discovery_inputs
    prepared = prepare_object_scene_semantic_registry_proposal(artifacts, roles)
    side0 = _aliases(prepared, 0)
    side1 = _aliases(prepared, 1)
    # These are the six operational cards returned by the historical v4
    # proposer. Citations are rebound only because aliases are local to the
    # deterministic synthetic discovery fixture.
    payload = {
        "side0_positive": [
            {
                "scope": "entity",
                "phrase": "two unequal opposing wedges joined at a narrow center",
                "required_witnesses": [
                    {
                        "kind": "count_relation",
                        "statement": (
                            "exactly two primary spreading portions belong to the entity"
                        ),
                    },
                    {
                        "kind": "spatial_relation",
                        "statement": (
                            "the two portions broaden into opposite regions from one shared narrow center"
                        ),
                    },
                    {
                        "kind": "shape_appearance",
                        "statement": (
                            "the two portions have visibly unequal extents or contours"
                        ),
                    },
                ],
                "accepted_variants": [
                    "fan-like, sector-like, and wedge-like portions count when each broadens away from the shared center"
                ],
                "near_miss_boundaries": [
                    "a three-armed radial cluster falls outside this concept",
                    "two portions meeting along a broad shared edge do not qualify",
                ],
                "citations": list(side0),
            },
            {
                "scope": "entity",
                "phrase": "circle and quadrilateral loops trace the same entity",
                "required_witnesses": [
                    {
                        "kind": "marking_pattern",
                        "statement": (
                            "multiple hollow circular loops visibly repeat along an entity boundary or path"
                        ),
                    },
                    {
                        "kind": "marking_pattern",
                        "statement": (
                            "multiple hollow four-sided loops visibly repeat along the same entity boundary or path"
                        ),
                    },
                    {
                        "kind": "part_topology",
                        "statement": (
                            "the circular and four-sided loop chains participate in one composite entity"
                        ),
                    },
                ],
                "accepted_variants": [
                    "round, oval, square, rectangular, and irregular four-sided hollow units count as boundary loops"
                ],
                "near_miss_boundaries": [
                    "solid dots and solid blocks fall outside the hollow-loop count",
                    "loops scattered through the interior rather than tracing paths are excluded",
                ],
                "citations": list(side0[:4]),
            },
            {
                "scope": "panel",
                "phrase": "one patterned figure and one plain outlined figure",
                "required_witnesses": [
                    {
                        "kind": "count_relation",
                        "statement": (
                            "exactly two spatially separate primary figures occupy the panel"
                        ),
                    },
                    {
                        "kind": "marking_pattern",
                        "statement": (
                            "one figure has a boundary or path assembled from many repeated small outlined units"
                        ),
                    },
                    {
                        "kind": "marking_pattern",
                        "statement": (
                            "the other figure is drawn predominantly with continuous plain outline strokes"
                        ),
                    },
                ],
                "accepted_variants": [
                    "a plain figure may use several continuous polygonal or curved strokes",
                    "the patterned figure may use circular, polygonal, scalloped, or beaded boundary units",
                ],
                "near_miss_boundaries": [
                    "a panel with repeated boundary units on both figures is excluded",
                    "decoration confined inside a plain outer boundary does not qualify as a patterned boundary",
                ],
                "citations": list(side0),
            },
        ],
        "side1_positive": [
            {
                "scope": "entity",
                "phrase": "three patterned arms radiate from a compact center",
                "required_witnesses": [
                    {
                        "kind": "count_relation",
                        "statement": (
                            "three primary elongated arms are visibly distinguishable within one entity"
                        ),
                    },
                    {
                        "kind": "spatial_relation",
                        "statement": (
                            "all three arms diverge from the same compact central meeting region"
                        ),
                    },
                    {
                        "kind": "marking_pattern",
                        "statement": (
                            "repeated small outlined geometric units occur along at least two arms"
                        ),
                    },
                ],
                "accepted_variants": [
                    "a broad curved wedge and narrow linear extensions count as arms when all radiate from one center",
                    "rows of repeated circles, triangles, or quadrilaterals count as patterned arms",
                ],
                "near_miss_boundaries": [
                    "two opposing lobes joined through a waist do not qualify",
                    "three chains forming a closed triangular frame are excluded",
                ],
                "citations": list(side1[:3]),
            },
            {
                "scope": "entity",
                "phrase": "open paths coexist with many closed symbol loops",
                "required_witnesses": [
                    {
                        "kind": "part_topology",
                        "statement": (
                            "at least one visible path in the entity terminates at a free endpoint"
                        ),
                    },
                    {
                        "kind": "count_relation",
                        "statement": (
                            "multiple small outlined symbols form closed loops within the same entity"
                        ),
                    },
                    {
                        "kind": "marking_pattern",
                        "statement": (
                            "the closed symbol loops repeat along one or more visible paths"
                        ),
                    },
                ],
                "accepted_variants": [
                    "visible stroke terminals and visibly unfinished patterned strands count as open paths",
                    "hollow circles, triangles, quadrilaterals, and irregular polygons count as closed symbol loops",
                ],
                "near_miss_boundaries": [
                    "a single open arc without repeated closed symbols falls outside this concept",
                    "separate open and looped figures do not qualify as one entity",
                ],
                "citations": list(side1[:4]),
            },
            {
                "scope": "panel",
                "phrase": "two separated figures both use repeated geometric units",
                "required_witnesses": [
                    {
                        "kind": "count_relation",
                        "statement": (
                            "exactly two spatially separate primary figures occupy the panel"
                        ),
                    },
                    {
                        "kind": "marking_pattern",
                        "statement": (
                            "one figure contains a visible chain of repeated small outlined geometric units"
                        ),
                    },
                    {
                        "kind": "marking_pattern",
                        "statement": (
                            "the other figure also contains a visible chain of repeated small outlined geometric units"
                        ),
                    },
                ],
                "accepted_variants": [
                    "circles, triangles, quadrilaterals, and irregular hollow polygons count as repeated geometric units",
                    "dense zigzags count as patterning when paired with repeated hollow geometric units",
                ],
                "near_miss_boundaries": [
                    "a panel with repeated units on only one figure is excluded",
                    "two plain continuous outlines fall outside this concept",
                ],
                "citations": list(side1[:3]),
            },
        ],
    }

    _, buckets, dropped = semantic_registry._project_semantic_payload(
        prepared, payload, require_usable_buckets=False
    )
    assert buckets[0] == []
    assert tuple(item.phrase for item in buckets[1]) == (
        "three patterned arms radiate from a compact center",
        "two separated figures both use repeated geometric units",
    )
    assert tuple(
        (item.orientation, item.input_index, item.reason_code)
        for item in dropped
    ) == (
        ("side0_positive", 0, "criteria_policy"),
        ("side0_positive", 1, "criteria_policy"),
        ("side0_positive", 2, "criteria_policy"),
        ("side1_positive", 1, "criteria_policy"),
    )


def test_orientation_is_audit_only_and_union_registry_is_role_blind(
    discovery_inputs,
):
    artifacts, roles = discovery_inputs
    prepared = prepare_object_scene_semantic_registry_proposal(artifacts, roles)
    payload = _valid_payload(prepared)
    _, registry = build_object_scene_semantic_registry_proposal(prepared, payload)

    swapped_roles = tuple(
        {**row, "historical_role": 1 - row["historical_role"]} for row in roles
    )
    swapped_prepared = prepare_object_scene_semantic_registry_proposal(
        artifacts, swapped_roles
    )
    swapped_payload = {
        "side0_positive": deepcopy(payload["side1_positive"]),
        "side1_positive": deepcopy(payload["side0_positive"]),
    }
    _, swapped_registry = build_object_scene_semantic_registry_proposal(
        swapped_prepared, swapped_payload
    )
    assert swapped_registry == registry
    evaluator_view = str(registry.to_data())
    assert "side0_positive" not in evaluator_view
    assert "side1_positive" not in evaluator_view
    assert "citations" not in evaluator_view


@pytest.mark.parametrize(
    "phrase",
    (
        "not pointed",
        "without internal marks",
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
def test_quarantines_negation_logical_packaging_and_role_comparison(
    discovery_inputs, phrase,
):
    artifacts, roles = discovery_inputs
    prepared = prepare_object_scene_semantic_registry_proposal(artifacts, roles)
    payload = _valid_payload(prepared)
    payload["side0_positive"][0]["phrase"] = phrase
    proposal, registry = build_object_scene_semantic_registry_proposal(
        prepared, payload
    )
    assert proposal.status == "proposed"
    assert len(proposal.dropped_concepts) == 1
    assert proposal.dropped_concepts[0].orientation == "side0_positive"
    assert proposal.dropped_concepts[0].input_index == 0
    assert proposal.dropped_concepts[0].reason_code == "phrase_policy"
    assert phrase not in {item.tag for item in registry.tags}
    assert {item.tag for item in registry.tags} == {
        "mismatched parts",
        "unequal edge lengths",
        "balanced spacing",
    }


def test_quarantines_duplicate_scoped_phrase_and_duplicate_citation(discovery_inputs):
    artifacts, roles = discovery_inputs
    prepared = prepare_object_scene_semantic_registry_proposal(artifacts, roles)
    payload = _valid_payload(prepared)
    payload["side1_positive"][1]["scope"] = "entity"
    payload["side1_positive"][1]["phrase"] = "mismatched parts"
    assert (
        payload["side1_positive"][1]["required_witnesses"]
        != payload["side0_positive"][1]["required_witnesses"]
    )
    proposal, registry = build_object_scene_semantic_registry_proposal(
        prepared, payload
    )
    assert tuple(item.reason_code for item in proposal.dropped_concepts) == (
        "duplicate_scoped_phrase",
        "duplicate_scoped_phrase",
    )
    assert "mismatched parts" not in {item.tag for item in registry.tags}

    payload = _valid_payload(prepared)
    alias = _aliases(prepared, 0)[0]
    payload["side0_positive"][0]["citations"] = [alias, alias]
    proposal, registry = build_object_scene_semantic_registry_proposal(
        prepared, payload
    )
    assert len(proposal.dropped_concepts) == 1
    assert proposal.dropped_concepts[0].reason_code == "citation_policy"
    assert "paired visible forms" not in {item.tag for item in registry.tags}


def test_quarantines_foreign_and_cross_side_citations(discovery_inputs):
    artifacts, roles = discovery_inputs
    prepared = prepare_object_scene_semantic_registry_proposal(artifacts, roles)
    for citation in ("panel_999", _aliases(prepared, 1)[0]):
        payload = _valid_payload(prepared)
        payload["side0_positive"][0]["citations"] = [
            _aliases(prepared, 0)[0], citation
        ]
        proposal, registry = build_object_scene_semantic_registry_proposal(
            prepared, payload
        )
        assert len(proposal.dropped_concepts) == 1
        assert proposal.dropped_concepts[0].reason_code == "foreign_citation"
        assert "paired visible forms" not in {item.tag for item in registry.tags}


def test_quarantines_malformed_citation_items_without_losing_valid_concepts(
    discovery_inputs,
):
    artifacts, roles = discovery_inputs
    prepared = prepare_object_scene_semantic_registry_proposal(artifacts, roles)
    for malformed in ([{}, {}], [1, 2], [_aliases(prepared, 0)[0], 2]):
        payload = _valid_payload(prepared)
        payload["side0_positive"][0]["citations"] = malformed
        proposal, registry = build_object_scene_semantic_registry_proposal(
            prepared, payload
        )
        assert len(proposal.dropped_concepts) == 1
        assert proposal.dropped_concepts[0].reason_code == "citation_policy"
        assert "paired visible forms" not in {item.tag for item in registry.tags}


def test_all_invalid_rows_in_one_bucket_produce_typed_payload_error(
    discovery_inputs,
):
    artifacts, roles = discovery_inputs
    prepared = prepare_object_scene_semantic_registry_proposal(artifacts, roles)
    payload = _valid_payload(prepared)
    for row in payload["side0_positive"]:
        row["phrase"] = "not a visible affirmative concept"
    with pytest.raises(
        ObjectSceneSemanticRegistryError, match="no usable concept"
    ):
        build_object_scene_semantic_registry_proposal(prepared, payload)


def test_long_spatial_and_compound_visual_phrases_survive(
    discovery_inputs,
):
    artifacts, roles = discovery_inputs
    prepared = prepare_object_scene_semantic_registry_proposal(artifacts, roles)
    side0 = _aliases(prepared, 0)
    side1 = _aliases(prepared, 1)
    payload = {
        "side0_positive": [
            _concept(
                "entity",
                "a jagged region paired with chains of outlined motifs",
                side0[:2],
            ),
            _concept(
                "panel",
                "a larger intricate figure in the lower-left region",
                side0[1:3],
            ),
        ],
        "side1_positive": [
            _concept(
                "entity", "serrated edging along decorated bands", side1[:2]
            ),
            _concept(
                "entity",
                "mixed circular and triangular marks",
                side1[1:3],
            ),
        ],
    }
    proposal, registry = build_object_scene_semantic_registry_proposal(
        prepared, payload
    )
    assert proposal.status == "proposed"
    assert proposal.dropped_concepts == ()
    assert {item.tag for item in registry.tags} == {
        "a jagged region paired with chains of outlined motifs",
        "a larger intricate figure in the lower-left region",
        "serrated edging along decorated bands",
        "mixed circular and triangular marks",
    }


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

    raw = proposal.to_data()
    raw["model_payload"]["side0_positive"][0]["required_witnesses"][0][
        "statement"
    ] = "a visibly altered witness statement"
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


def test_orientation_coverage_gap_binds_valid_rows_and_quarantines_invalid_rows(
    discovery_inputs,
):
    artifacts, roles = discovery_inputs
    prepared = prepare_object_scene_semantic_registry_proposal(artifacts, roles)
    payload = {
        "side0_positive": [
            _concept(
                "entity", "a birdlike angular silhouette", _aliases(prepared, 0)[:2]
            )
        ],
        "side1_positive": [
            _concept(
                "entity", "circular or triangular outlines", _aliases(prepared, 1)[:2]
            )
        ],
    }
    with pytest.raises(ObjectSceneSemanticRegistryError, match="no usable concept"):
        build_object_scene_semantic_registry_proposal(prepared, payload)
    proposal, registry = build_object_scene_semantic_registry_gap(
        prepared, "payload_rejected", payload
    )
    assert proposal.status == "typed_proposal_gap"
    assert proposal.model_payload == payload
    assert proposal.side0_positive == proposal.side1_positive == ()
    assert tuple(
        (item.orientation, item.input_index, item.reason_code)
        for item in proposal.dropped_concepts
    ) == (("side1_positive", 0, "phrase_policy"),)
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

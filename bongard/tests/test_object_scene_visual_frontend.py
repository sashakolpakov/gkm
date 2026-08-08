from __future__ import annotations

import ast
from copy import deepcopy
from dataclasses import replace
import hashlib
from io import BytesIO
import json
from pathlib import Path
import re

from PIL import Image, ImageDraw
import pytest

from bongard.evidence import Disposition
import bongard.object_scene_visual_frontend as frontend
from bongard.object_scene_visual_frontend import (
    OBJECT_SCENE_COUNT_OBSERVABLE_IDS,
    OBJECT_SCENE_MAX_TAG_CHARACTERS,
    OBJECT_SCENE_QUALITATIVE_OBSERVABLE_IDS,
    ObjectSceneProposalInventory,
    ObjectSceneSoftTagRegistry,
    ObjectSceneTranscriptArtifact,
    ObjectSceneTranscriptMode,
    ObjectSceneVisualFrontendError,
    extract_object_scene_proposal_inventory,
    freeze_object_scene_soft_tag_registry,
    lookup_object_scene_panel_soft_tag,
    lookup_object_scene_soft_tag,
    object_scene_transcript_output_schema,
    object_scene_transcript_prompt,
    observe_object_scene_transcript,
    prepare_object_scene_transcript_inputs,
    render_object_scene_proposal_atlas,
    verify_object_scene_proposal_inventory,
    verify_object_scene_soft_tag_registry,
    verify_object_scene_transcript_artifact,
)
from bongard.prototype_scene_observer import PrototypeSceneObserverStatus
from bongard.tests.test_prototype_object_lineages import _png as lineage_png
from bongard.tests.test_prototype_scene_observer import (
    EFFORT,
    LAUNCHER_DIGEST,
    MODEL,
    MODEL_CATALOG,
    NO_TOOLS_ATTESTATION,
    _receipt,
)
from bongard.transport import CodexStructuredResult


CONTEXT_A = "sha256:" + "a" * 64
CONTEXT_B = "sha256:" + "b" * 64


def _scene(shift: int = 0) -> bytes:
    image = Image.new("RGB", (96, 64), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((8 + shift, 12, 26 + shift, 34), fill="black")
    draw.ellipse((60, 16, 78, 34), fill="black")
    output = BytesIO()
    image.save(output, format="PNG", optimize=False)
    return output.getvalue()


def _overlap_scene() -> bytes:
    image = Image.new("RGB", (96, 64), "white")
    draw = ImageDraw.Draw(image)
    for box in ((10, 10, 18, 18), (25, 10, 33, 18), (75, 10, 83, 18)):
        draw.rectangle(box, fill="black")
    output = BytesIO()
    image.save(output, format="PNG", optimize=False)
    return output.getvalue()


def _blank_scene(width: int = 96) -> bytes:
    image = Image.new("RGB", (width, 64), "white")
    output = BytesIO()
    image.save(output, format="PNG", optimize=False)
    return output.getvalue()


def _registered_tag_payload(tag, *, state, evidence):
    return {
        "tag_id": tag.tag_id,
        "witness_cells": [
            {
                "witness_id": witness.witness_id,
                "state": state,
                "evidence": evidence,
            }
            for witness in tag.required_witnesses
        ],
    }


def _registry_with_tags(registry, tags):
    values = {
        "source_transcript_digests": registry.source_transcript_digests,
        "source_panel_digests": registry.source_panel_digests,
        "tags": tuple(tags),
        "dropped_tags": registry.dropped_tags,
    }
    provisional = object.__new__(ObjectSceneSoftTagRegistry)
    for name, value in values.items():
        object.__setattr__(provisional, name, value)
    return ObjectSceneSoftTagRegistry(
        **values,
        registry_digest=frontend.canonical_digest(
            frontend._registry_content(provisional)
        ),
    )


def _payload(
    inventory,
    *,
    open_tags=("bird-like object",),
    panel_open_tags=(),
    registry=None,
):
    rows = []
    panel_registry_tags = (
        ()
        if registry is None
        else tuple(item for item in registry.tags if item.scope == "panel")
    )
    entity_registry_tags = (
        ()
        if registry is None
        else tuple(item for item in registry.tags if item.scope == "entity")
    )
    for crop in inventory.objects:
        rows.append(
            {
                "object_id": crop.object_id,
                "summary": "outlined visible form",
                "counts": [
                    {
                        "observable_id": observable_id,
                        "state": "measured",
                        "lower_count": 0,
                        "upper_count": 2,
                        "evidence": "visible marks were inspected",
                    }
                    for observable_id in OBJECT_SCENE_COUNT_OBSERVABLE_IDS
                ],
                "observables": [
                    {
                        "observable_id": observable_id,
                        "state": "present" if observable_id == "bird_like" else "absent",
                        "evidence": "feature is visibly supported" if observable_id == "bird_like" else "feature is not visible",
                    }
                    for observable_id in OBJECT_SCENE_QUALITATIVE_OBSERVABLE_IDS
                ],
                "open_tags": [
                    {
                        "tag": tag,
                        "state": "present",
                        "evidence": "visible silhouette supports phrase",
                    }
                    for tag in sorted(open_tags)
                ] if registry is None else [],
                "registered_tags": [] if registry is None else [
                    _registered_tag_payload(
                        tag,
                        state=(
                            "present"
                            if tag.tag == "bird-like object"
                            else "indeterminate"
                        ),
                        evidence=(
                            "visible silhouette supports phrase"
                            if tag.tag == "bird-like object"
                            else "visible detail is unresolved"
                        ),
                    )
                    for tag in entity_registry_tags
                ],
            }
        )
    return {
        "panel": {
            "summary": "two visible forms occupy separated panel regions",
            "open_tags": [
                {
                    "tag": tag,
                    "state": "present",
                    "evidence": "complete composition supports phrase",
                }
                for tag in sorted(panel_open_tags)
            ] if registry is None else [],
            "registered_tags": [] if registry is None else [
                _registered_tag_payload(
                    tag,
                    state="present",
                    evidence="complete composition supports phrase",
                )
                for tag in panel_registry_tags
            ],
        },
        "objects": rows,
    }


def _transport(payload, calls, envelopes=None):
    def invoke(prompt, paths, names, schema, **kwargs):
        calls.append((tuple(names), kwargs))
        if envelopes is not None:
            envelopes.append((prompt, deepcopy(schema), tuple(names)))
        return CodexStructuredResult(payload, _receipt(prompt, paths, names, schema, payload))

    return invoke


def _observe(raw, payload, *, context=CONTEXT_A, mode=ObjectSceneTranscriptMode.DISCOVERY, registry=None, calls=None, envelopes=None):
    values = [] if calls is None else calls
    return observe_object_scene_transcript(
        raw,
        scene_id="opaque-scene",
        observation_context_digest=context,
        mode=mode,
        registry=registry,
        model=MODEL,
        reasoning_effort=EFFORT,
        expected_launcher_digest=LAUNCHER_DIGEST,
        model_catalog_snapshot=MODEL_CATALOG,
        no_tools_attestation=NO_TOOLS_ATTESTATION,
        transport=_transport(payload, values, envelopes),
    )


def test_inventory_is_deterministic_replayable_and_diagnostics_are_nondecisional():
    raw = lineage_png("stable_plus_threshold_only")
    inventory = extract_object_scene_proposal_inventory(raw)

    assert inventory.inventory_status == "complete"
    assert inventory.catalog_complete_under_rule is True
    assert inventory.diagnostic_codes == ("unlinked_hypotheses",)
    assert len(inventory.objects) == 1
    assert inventory.objects[0].object_id == "object_0000"
    assert inventory.objects[0].geometry_cells()["component_count"] == 1
    assert ObjectSceneProposalInventory.from_data(inventory.to_data()) == inventory
    atlas = dict(render_object_scene_proposal_atlas(inventory, raw))
    assert verify_object_scene_proposal_inventory(
        inventory, raw, expected_atlas_png_by_name=atlas
    ) == inventory


def test_overlap_graph_preserves_competing_proposals_without_semantic_claim():
    raw = _overlap_scene()
    inventory = extract_object_scene_proposal_inventory(raw)

    assert inventory.catalog_complete_under_rule is True
    assert len(inventory.objects) == 4
    assert inventory.objects[0].overlap_object_ids == ("object_0003",)
    assert inventory.objects[1].overlap_object_ids == ("object_0003",)
    assert inventory.objects[2].overlap_object_ids == ()
    assert inventory.objects[3].overlap_object_ids == ("object_0000", "object_0001")
    assert inventory.to_data()["semantic_object_completeness_claimed"] is False


def test_visible_envelope_is_opaque_and_exhausts_fixed_vocabulary():
    inventory = extract_object_scene_proposal_inventory(_scene())
    prompt = object_scene_transcript_prompt(inventory, ObjectSceneTranscriptMode.DISCOVERY)
    schema = object_scene_transcript_output_schema(inventory, ObjectSceneTranscriptMode.DISCOVERY)
    envelope = prompt + json.dumps(schema, sort_keys=True) + "panel.png" + "".join(item.name for item in inventory.atlas_sheets)

    for word in ("candidate", "group", "class", "label", "target", "foil", "predicate", "formula", "query", "positive", "negative"):
        assert re.search(rf"\b{word}s?\b", envelope, re.IGNORECASE) is None
    for crop in inventory.objects:
        assert f"{crop.object_id}: {crop.atlas_name}, row {crop.row_index}, column {crop.column_index}" in prompt
    for observable_id in (*OBJECT_SCENE_COUNT_OBSERVABLE_IDS, *OBJECT_SCENE_QUALITATIVE_OBSERVABLE_IDS):
        assert observable_id in prompt
    assert "Omission means only unrecorded and remains indeterminate" in prompt
    assert f"at most {OBJECT_SCENE_MAX_TAG_CHARACTERS} characters" in prompt
    tag_schema = schema["properties"]["panel"]["properties"]["open_tags"][
        "items"
    ]["properties"]["tag"]
    assert tag_schema["description"].endswith(
        f"at most {OBJECT_SCENE_MAX_TAG_CHARACTERS} characters."
    )
    assert "maxLength" not in tag_schema


def test_discovery_is_one_call_typed_and_cold_replayable():
    raw = _scene()
    inventory = extract_object_scene_proposal_inventory(raw)
    payload = _payload(inventory)
    calls = []
    artifact = _observe(raw, payload, calls=calls)

    assert artifact.status is PrototypeSceneObserverStatus.SUCCESS
    assert artifact.physical_call_count == 1
    assert len(calls) == 1
    assert calls[0][0] == ("panel.png", "objects_000.png")
    assert artifact.transcript is not None
    row = artifact.transcript.objects[0]
    assert row.qualitative_cells[OBJECT_SCENE_QUALITATIVE_OBSERVABLE_IDS.index("bird_like")].support.to_data() == {"lower": 1, "upper": 1}
    assert row.qualitative_cells[0].support.to_data() == {"lower": 0, "upper": 0}
    assert row.count_cells[0].interval.to_data() == {"lower": 0, "upper": 2}
    omitted = lookup_object_scene_soft_tag(artifact, "object_0000", "sector form")
    assert omitted.disposition is Disposition.INDETERMINATE
    assert omitted.support.to_data() == {"lower": 0, "upper": 1}
    assert ObjectSceneTranscriptArtifact.from_data(artifact.to_data()) == artifact
    assert verify_object_scene_transcript_artifact(
        artifact,
        raw,
        expected_scene_id="opaque-scene",
        expected_observation_context_digest=CONTEXT_A,
        expected_panel_sha256=hashlib.sha256(raw).hexdigest(),
        expected_artifact_digest=artifact.artifact_digest,
    ) == artifact


def test_panel_discovery_is_typed_scoped_and_distinct_from_entity_discovery():
    raw_a, raw_b = _scene(0), _scene(2)
    inv_a = extract_object_scene_proposal_inventory(raw_a)
    inv_b = extract_object_scene_proposal_inventory(raw_b)
    phrase = "paired visible forms"
    first = _observe(
        raw_a,
        _payload(inv_a, open_tags=(phrase,), panel_open_tags=(phrase,)),
    )
    second = _observe(
        raw_b,
        _payload(inv_b, open_tags=(phrase,), panel_open_tags=(phrase,)),
        context=CONTEXT_B,
    )

    assert first.status is second.status is PrototypeSceneObserverStatus.SUCCESS
    assert first.transcript is not None and second.transcript is not None
    assert first.transcript.panel_summary == (
        "two visible forms occupy separated panel regions"
    )
    assert first.transcript.panel_open_tags[0].tag == phrase
    assert lookup_object_scene_panel_soft_tag(first, phrase).disposition is (
        Disposition.PRESENT
    )
    assert lookup_object_scene_panel_soft_tag(
        first, "diagonal arrangement"
    ).disposition is Disposition.INDETERMINATE

    registry = freeze_object_scene_soft_tag_registry(
        (first.transcript, second.transcript)
    )
    assert tuple(
        (item.tag_id, item.scope, item.tag, item.distinct_panel_count)
        for item in registry.tags
    ) == (
        ("tag_0000", "entity", phrase, 2),
        ("tag_0001", "panel", phrase, 2),
    )
    assert registry.tags[0].tag_digest != registry.tags[1].tag_digest


def test_model_unicode_punctuation_is_normalized_before_persistence_and_replay():
    raw = _scene()
    inventory = extract_object_scene_proposal_inventory(raw)
    payload = _payload(inventory)
    payload["panel"]["summary"] = "two forms—side by side"
    payload["panel"]["open_tags"] = [
        {
            "tag": "side—by—side forms",
            "state": "present",
            "evidence": "forms appear—side by side",
        }
    ]
    for row in payload["objects"]:
        row["summary"] = "outlined form—leans left"
        row["counts"][0]["evidence"] = "visible marks–two"
        row["observables"][0]["evidence"] = "edge—clearly visible"
        row["open_tags"] = [
            {
                "tag": "wing—like form",
                "state": "present",
                "evidence": "wing—like contour is visible",
            }
        ]

    artifact = _observe(raw, payload)

    assert artifact.status is PrototypeSceneObserverStatus.SUCCESS
    assert artifact.transcript is not None
    row = artifact.transcript.objects[0]
    assert artifact.transcript.panel_summary == "two forms - side by side"
    assert artifact.transcript.panel_open_tags[0].tag == "side - by - side forms"
    assert artifact.transcript.panel_open_tags[0].evidence == (
        "forms appear - side by side"
    )
    assert row.summary == "outlined form - leans left"
    assert row.count_cells[0].evidence == "visible marks - two"
    assert row.qualitative_cells[0].evidence == "edge - clearly visible"
    assert row.open_tags[0].tag == "wing - like form"
    assert row.open_tags[0].evidence == "wing - like contour is visible"
    assert "–" not in str(artifact.transcript.to_data())
    assert "—" not in str(artifact.transcript.to_data())
    assert ObjectSceneTranscriptArtifact.from_data(artifact.to_data()) == artifact
    assert verify_object_scene_transcript_artifact(
        artifact,
        raw,
        expected_scene_id="opaque-scene",
        expected_observation_context_digest=CONTEXT_A,
        expected_panel_sha256=hashlib.sha256(raw).hexdigest(),
        expected_artifact_digest=artifact.artifact_digest,
    ) == artifact


def test_ordinary_lean_language_is_not_confused_with_proof_assistant_language():
    raw = _scene()
    inventory = extract_object_scene_proposal_inventory(raw)
    accepted = _payload(inventory)
    for row in accepted["objects"]:
        row["summary"] = "outlined form leans left"
    artifact = _observe(raw, accepted)
    assert artifact.status is PrototypeSceneObserverStatus.SUCCESS
    assert artifact.transcript is not None
    assert artifact.transcript.objects[0].summary == "outlined form leans left"

    standalone = _payload(inventory)
    for row in standalone["objects"]:
        row["summary"] = "lean bird-like silhouette"
    standalone_artifact = _observe(raw, standalone)
    assert standalone_artifact.status is PrototypeSceneObserverStatus.SUCCESS
    assert standalone_artifact.transcript is not None
    assert standalone_artifact.transcript.objects[0].summary == (
        "lean bird-like silhouette"
    )

    forbidden = _payload(inventory)
    forbidden["objects"][0]["summary"] = "Lean proof appears here"
    rejected = _observe(raw, forbidden)
    assert rejected.status is PrototypeSceneObserverStatus.PARSER_ERROR
    assert rejected.transcript is None


def test_discovery_open_tags_are_sorted_and_all_duplicate_members_are_quarantined():
    raw = _scene()
    inventory = extract_object_scene_proposal_inventory(raw)
    payload = _payload(inventory, open_tags=("arched form", "wing-like form"))
    for row in payload["objects"]:
        row["open_tags"].reverse()
    reversed_artifact = _observe(raw, payload)
    assert reversed_artifact.status is PrototypeSceneObserverStatus.SUCCESS
    assert reversed_artifact.transcript is not None
    assert tuple(item.tag for item in reversed_artifact.transcript.objects[0].open_tags) == (
        "arched form",
        "wing-like form",
    )

    ordered_payload = _payload(
        inventory, open_tags=("arched form", "wing-like form")
    )
    ordered_artifact = _observe(raw, ordered_payload)
    assert ordered_artifact.status is PrototypeSceneObserverStatus.SUCCESS
    assert ordered_artifact.transcript == reversed_artifact.transcript
    assert ObjectSceneTranscriptArtifact.from_data(
        reversed_artifact.to_data()
    ) == reversed_artifact

    duplicate = deepcopy(payload)
    duplicate["objects"][0]["open_tags"] = [
        {
            "tag": "wing—like form",
            "state": "present",
            "evidence": "visible wing contour",
        },
        {
            "tag": "wing - like form",
            "state": "present",
            "evidence": "same visible wing contour",
        },
    ]
    salvaged = _observe(raw, duplicate)
    assert salvaged.status is PrototypeSceneObserverStatus.SUCCESS
    assert salvaged.transcript is not None
    assert salvaged.transcript.objects[0].open_tags == ()
    drops = tuple(
        item
        for item in salvaged.transcript.dropped_open_tags
        if item.owner_id == "object_0000"
    )
    assert tuple(item.reason_code for item in drops) == (
        "duplicate_normalized_tag",
        "duplicate_normalized_tag",
    )
    assert tuple(item.item_index for item in drops) == (0, 1)
    assert tuple(item.raw_item_digest for item in drops) == tuple(
        frontend.canonical_digest(item)
        for item in duplicate["objects"][0]["open_tags"]
    )
    assert lookup_object_scene_soft_tag(
        salvaged, "object_0000", "wing - like form"
    ).disposition is Disposition.INDETERMINATE


def test_discovery_accepts_different_and_other_as_affirmative_visual_words():
    raw = _scene()
    inventory = extract_object_scene_proposal_inventory(raw)
    artifact = _observe(
        raw,
        _payload(
            inventory,
            open_tags=("different sized arcs", "other visible form"),
        ),
    )

    assert artifact.status is PrototypeSceneObserverStatus.SUCCESS
    assert artifact.transcript is not None
    assert tuple(
        item.tag for item in artifact.transcript.objects[0].open_tags
    ) == ("different sized arcs", "other visible form")


def test_bad_discovery_tag_items_preserve_summaries_fixed_cells_and_cold_replay():
    raw = _scene()
    inventory = extract_object_scene_proposal_inventory(raw)
    payload = _payload(inventory, open_tags=())
    payload["panel"]["open_tags"] = [
        {
            "tag": "x" * (OBJECT_SCENE_MAX_TAG_CHARACTERS + 1),
            "state": "present",
            "evidence": "complete composition contains a visible form",
        }
    ]
    row = payload["objects"][0]
    row["open_tags"] = [
        {
            "tag": "larger upper arc and smaller lower arc",
            "state": "present",
            "evidence": "two differently sized arcs are visible",
        },
        {
            "tag": "left arm longer than right arm",
            "state": "present",
            "evidence": "the left arm visibly extends farther",
        },
        {
            "tag": "shape without internal marks",
            "state": "present",
            "evidence": "the interior appears plain",
        },
        {
            "tag": "circle or triangle",
            "state": "indeterminate",
            "evidence": "the outline is visually ambiguous",
        },
        {
            "tag": "role marked form",
            "state": "present",
            "evidence": "an outlined form is visible",
        },
        {
            "tag": "y" * (OBJECT_SCENE_MAX_TAG_CHARACTERS + 1),
            "state": "present",
            "evidence": "an elongated form is visible",
        },
        "not-an-object",
        {"tag": "missing evidence form", "state": "present"},
        {
            "tag": "pointed state form",
            "state": "absent",
            "evidence": "a pointed contour was inspected",
        },
        {
            "tag": "small evidence form",
            "state": "present",
            "evidence": "x",
        },
    ]

    artifact = _observe(raw, payload)

    assert artifact.status is PrototypeSceneObserverStatus.SUCCESS
    assert artifact.transcript is not None
    transcript = artifact.transcript
    assert transcript.panel_summary == payload["panel"]["summary"]
    assert transcript.panel_open_tags == ()
    assert transcript.objects[0].summary == row["summary"]
    assert len(transcript.objects[0].count_cells) == len(
        OBJECT_SCENE_COUNT_OBSERVABLE_IDS
    )
    assert len(transcript.objects[0].qualitative_cells) == len(
        OBJECT_SCENE_QUALITATIVE_OBSERVABLE_IDS
    )
    assert tuple(item.tag for item in transcript.objects[0].open_tags) == tuple(
        sorted(
            (
                "larger upper arc and smaller lower arc",
                "left arm longer than right arm",
            )
        )
    )
    assert tuple(item.reason_code for item in transcript.dropped_open_tags) == (
        "invalid_tag",
        "invalid_tag",
        "invalid_tag",
        "invalid_tag",
        "invalid_tag",
        "malformed_item",
        "malformed_item",
        "invalid_state",
        "invalid_evidence",
    )
    assert transcript.dropped_open_tags[0].scope == "panel"
    assert transcript.dropped_open_tags[0].owner_id == "panel"
    assert transcript.dropped_open_tags[0].item_index == 0
    assert transcript.dropped_open_tags[0].raw_item_digest == (
        frontend.canonical_digest(payload["panel"]["open_tags"][0])
    )
    assert verify_object_scene_transcript_artifact(
        artifact,
        raw,
        expected_scene_id="opaque-scene",
        expected_observation_context_digest=CONTEXT_A,
        expected_panel_sha256=hashlib.sha256(raw).hexdigest(),
        expected_artifact_digest=artifact.artifact_digest,
    ) == artifact

    tampered = deepcopy(artifact.to_data())
    dropped = tampered["transcript"]["dropped_open_tags"][0]
    dropped["raw_item_digest"] = "f" * 64
    dropped["drop_digest"] = frontend.canonical_digest(
        {key: value for key, value in dropped.items() if key != "drop_digest"}
    )
    transcript_data = tampered["transcript"]
    transcript_data["transcript_digest"] = frontend.canonical_digest(
        {
            key: value
            for key, value in transcript_data.items()
            if key != "transcript_digest"
        }
    )
    tampered["artifact_digest"] = frontend.canonical_digest(
        {key: value for key, value in tampered.items() if key != "artifact_digest"}
    )
    resealed = ObjectSceneTranscriptArtifact.from_data(tampered)
    with pytest.raises(
        ObjectSceneVisualFrontendError,
        match="transcript replay differs",
    ):
        verify_object_scene_transcript_artifact(
            resealed,
            raw,
            expected_scene_id="opaque-scene",
            expected_observation_context_digest=CONTEXT_A,
            expected_panel_sha256=hashlib.sha256(raw).hexdigest(),
            expected_artifact_digest=resealed.artifact_digest,
        )


def test_discovery_tag_capacity_is_lexical_and_excess_items_are_bound():
    raw = _scene()
    inventory = extract_object_scene_proposal_inventory(raw)
    payload = _payload(inventory, open_tags=())
    phrases = (
        "form alpha",
        "form beta",
        "form gamma",
        "form delta",
        "form epsilon",
        "form zeta",
        "form eta",
        "form theta",
        "form iota",
        "form kappa",
    )
    raw_items = [
        {
            "tag": phrase,
            "state": "present",
            "evidence": "the outlined form supports this phrase",
        }
        for phrase in reversed(phrases)
    ]
    payload["objects"][0]["open_tags"] = raw_items

    artifact = _observe(raw, payload)

    assert artifact.status is PrototypeSceneObserverStatus.SUCCESS
    assert artifact.transcript is not None
    assert tuple(item.tag for item in artifact.transcript.objects[0].open_tags) == (
        tuple(sorted(phrases))[:8]
    )
    capacity_drops = tuple(
        item
        for item in artifact.transcript.dropped_open_tags
        if item.owner_id == "object_0000"
    )
    assert tuple(item.reason_code for item in capacity_drops) == (
        "capacity_exceeded",
        "capacity_exceeded",
    )
    assert {
        item.raw_item_digest for item in capacity_drops
    } == {
        frontend.canonical_digest(item)
        for item in raw_items
        if item["tag"] in set(sorted(phrases)[8:])
    }


def test_registry_uses_distinct_panel_frequency_and_persists_every_drop():
    raw_a, raw_b = _scene(0), _scene(2)
    inventory_a = extract_object_scene_proposal_inventory(raw_a)
    inventory_b = extract_object_scene_proposal_inventory(raw_b)
    discovery_a = _observe(raw_a, _payload(inventory_a, open_tags=("bird-like object", "pointed form")))
    discovery_b = _observe(raw_b, _payload(inventory_b, open_tags=("bird-like object", "sector form")), context=CONTEXT_B)
    assert discovery_a.transcript is not None and discovery_b.transcript is not None

    registry = freeze_object_scene_soft_tag_registry((discovery_a.transcript, discovery_b.transcript))
    assert tuple((item.tag_id, item.scope, item.tag, item.distinct_panel_count) for item in registry.tags) == (("tag_0000", "entity", "bird-like object", 2),)
    assert tuple((item.scope, item.tag, item.reason) for item in registry.dropped_tags) == (
        ("entity", "pointed form", "seen_on_fewer_than_2_panels"),
        ("entity", "sector form", "seen_on_fewer_than_2_panels"),
    )
    assert ObjectSceneSoftTagRegistry.from_data(registry.to_data()) == registry
    assert verify_object_scene_soft_tag_registry(
        registry,
        (discovery_a.transcript, discovery_b.transcript),
        expected_registry_digest=registry.registry_digest,
    ) == registry


def test_operational_card_is_canonical_hashed_and_rendered_per_witness():
    raw_a, raw_b = _scene(0), _scene(2)
    inventory_a = extract_object_scene_proposal_inventory(raw_a)
    inventory_b = extract_object_scene_proposal_inventory(raw_b)
    first = _observe(raw_a, _payload(inventory_a))
    second = _observe(raw_b, _payload(inventory_b), context=CONTEXT_B)
    base = freeze_object_scene_soft_tag_registry(
        (first.transcript, second.transcript)
    )
    original = base.tags[0]
    tag = frontend.ObjectSceneSoftTag.create(
        original.tag_id,
        original.scope,
        original.tag,
        original.distinct_panel_count,
        (
            {
                "kind": "shape_appearance",
                "statement": "a compact body has pointed outer tips",
            },
                {
                    "kind": "part_topology",
                    "statement": "two lateral parts join one central body",
                },
                {
                    "kind": "spatial_relation",
                    "statement": "the lateral parts extend in opposite directions",
            },
        ),
        ("rounded outer tips also qualify",),
        ("a single rounded lobe does not qualify",),
    )
    registry = _registry_with_tags(base, (tag,))

    assert tuple(
        (item.witness_id, item.kind, item.statement)
        for item in tag.required_witnesses
    ) == (
        (
            "witness_00",
            "part_topology",
            "two lateral parts join one central body",
        ),
        (
            "witness_01",
            "shape_appearance",
            "a compact body has pointed outer tips",
        ),
        (
            "witness_02",
            "spatial_relation",
            "the lateral parts extend in opposite directions",
        ),
    )
    assert frontend.ObjectSceneSoftTag.from_data(tag.to_data()) == tag
    assert ObjectSceneSoftTagRegistry.from_data(registry.to_data()) == registry
    assert tag.tag_digest != original.tag_digest

    prompt = object_scene_transcript_prompt(
        inventory_a,
        ObjectSceneTranscriptMode.REGISTERED_EVALUATION,
        registry,
    )
    for witness in tag.required_witnesses:
        assert (
            f"{witness.witness_id} [{witness.kind}]: {witness.statement}"
            in prompt
        )
    assert "rounded outer tips also qualify" in prompt
    assert "a single rounded lobe does not qualify" in prompt
    assert "neither casts a vote" in prompt
    assert "Do not return a combined tag state" in prompt
    assert re.search(
        r"\b(?:citation|orientation|role)s?\b", prompt, re.IGNORECASE
    ) is None

    schema = object_scene_transcript_output_schema(
        inventory_a,
        ObjectSceneTranscriptMode.REGISTERED_EVALUATION,
        registry,
    )
    registered = schema["properties"]["objects"]["items"]["properties"][
        "registered_tags"
    ]["items"]
    assert set(registered["properties"]) == {"tag_id", "witness_cells"}
    witness_schema = registered["properties"]["witness_cells"]["items"]
    assert set(witness_schema["properties"]) == {
        "witness_id",
        "state",
        "evidence",
    }

    tampered = deepcopy(tag.to_data())
    tampered["required_witnesses"][0]["statement"] = (
        "two lateral parts touch one central body"
    )
    with pytest.raises(ObjectSceneVisualFrontendError):
        frontend.ObjectSceneSoftTag.from_data(tampered)


def test_operational_prose_admits_visual_paths_and_variant_alternatives_only():
    assert frontend.normalize_object_scene_witness_statement(
        "one visible path follows the left side"
    ) == "one visible path follows the left side"
    assert frontend.normalize_object_scene_accepted_variant(
        "round, oval, square, or irregular loops count as boundary units"
    ) == "round, oval, square, or irregular loops count as boundary units"
    assert frontend.normalize_object_scene_near_miss_boundary(
        "a path confined to one side falls outside this concept"
    ) == "a path confined to one side falls outside this concept"

    with pytest.raises(ObjectSceneVisualFrontendError):
        frontend.normalize_object_scene_witness_statement(
            "the portions differ in extent or contour"
        )
    with pytest.raises(ObjectSceneVisualFrontendError):
        frontend.normalize_object_scene_witness_statement(
            "the entity has round, oval, and square boundary units"
        )
    with pytest.raises(ObjectSceneVisualFrontendError):
        frontend.normalize_object_scene_accepted_variant(
            "round,oval, or square loops count as boundary units"
        )
    with pytest.raises(ObjectSceneVisualFrontendError):
        frontend.normalize_object_scene_accepted_variant(
            "round loops do not qualify"
        )
    with pytest.raises(ObjectSceneVisualFrontendError):
        frontend.normalize_object_scene_accepted_variant(
            "either round or square loops count as boundary units"
        )
    with pytest.raises(ObjectSceneVisualFrontendError):
        frontend.normalize_object_scene_near_miss_boundary(
            "a path without markers falls outside this concept"
        )
    with pytest.raises(ObjectSceneVisualFrontendError):
        frontend.normalize_object_scene_near_miss_boundary(
            "a path is not visible and one loop is excluded"
        )


def test_registered_prompt_admits_rigorous_visual_path_and_side_cards():
    raw_a, raw_b = _scene(0), _scene(2)
    inventory_a = extract_object_scene_proposal_inventory(raw_a)
    inventory_b = extract_object_scene_proposal_inventory(raw_b)
    first = _observe(raw_a, _payload(inventory_a))
    second = _observe(raw_b, _payload(inventory_b), context=CONTEXT_B)
    base = freeze_object_scene_soft_tag_registry(
        (first.transcript, second.transcript)
    )
    original = base.tags[0]
    tag = frontend.ObjectSceneSoftTag.create(
        original.tag_id,
        original.scope,
        original.tag,
        original.distinct_panel_count,
        (
            {
                "kind": "marking_pattern",
                "statement": "repeated hollow markers trace one visible path",
            },
            {
                "kind": "spatial_relation",
                "statement": "two visible paths cross from opposite sides",
            },
        ),
        ("overlap, intersection, or through-crossing counts as path crossing",),
        ("paths touching only at terminal tips are excluded",),
    )
    registry = _registry_with_tags(base, (tag,))

    prepared = prepare_object_scene_transcript_inputs(
        raw_a,
        inventory_a,
        ObjectSceneTranscriptMode.REGISTERED_EVALUATION,
        registry,
    )

    assert "two visible paths cross from opposite sides" in prepared.prompt
    assert "through-crossing counts as path crossing" in prepared.prompt


def test_registered_witness_macro_has_closed_precedence_and_error_replay():
    tag = frontend.ObjectSceneSoftTag.create(
        "tag_0000",
        "entity",
        "bird-like object",
        2,
        (
            {
                "kind": "part_topology",
                "statement": "two lateral parts join one central body",
            },
            {
                "kind": "shape_appearance",
                "statement": "a compact body has pointed outer tips",
            },
            {
                "kind": "spatial_relation",
                "statement": "the lateral parts extend in opposite directions",
            },
        ),
    )

    def compiled(states):
        cells = []
        for witness, state in zip(tag.required_witnesses, states, strict=True):
            if state == "error":
                cells.append(
                    frontend.ObjectSceneRegisteredWitnessCell.error(
                        witness, "witness evaluation is unavailable"
                    )
                )
            else:
                cells.append(
                    frontend.ObjectSceneRegisteredWitnessCell.create(
                        witness, state, "visible detail was inspected"
                    )
                )
        return frontend.ObjectSceneRegisteredTagCell.from_witness_cells(
            tag, cells
        )

    error = compiled(("absent", "error", "present"))
    absent = compiled(("present", "absent", "indeterminate"))
    present = compiled(("present", "present", "present"))
    indeterminate = compiled(("present", "indeterminate", "present"))

    assert error.disposition is Disposition.ERROR
    assert error.support is None
    assert absent.disposition is Disposition.CERTIFIED_ABSENT
    assert absent.support.to_data() == {"lower": 0, "upper": 0}
    assert present.disposition is Disposition.PRESENT
    assert present.support.to_data() == {"lower": 1, "upper": 1}
    assert indeterminate.disposition is Disposition.INDETERMINATE
    assert indeterminate.support.to_data() == {"lower": 0, "upper": 1}
    assert (
        frontend.ObjectSceneRegisteredTagCell.from_data(error.to_data())
        == error
    )
    assert error.to_data()["state"] == "error"


def test_registered_witness_payload_is_exact_and_macro_is_python_compiled():
    raw_a, raw_b = _scene(0), _scene(2)
    inventory_a = extract_object_scene_proposal_inventory(raw_a)
    inventory_b = extract_object_scene_proposal_inventory(raw_b)
    first = _observe(raw_a, _payload(inventory_a))
    second = _observe(raw_b, _payload(inventory_b), context=CONTEXT_B)
    base = freeze_object_scene_soft_tag_registry(
        (first.transcript, second.transcript)
    )
    original = base.tags[0]
    tag = frontend.ObjectSceneSoftTag.create(
        original.tag_id,
        original.scope,
        original.tag,
        original.distinct_panel_count,
        (
            {
                "kind": "shape_appearance",
                "statement": "a compact body has pointed outer tips",
            },
            {
                "kind": "part_topology",
                "statement": "two lateral parts join one central body",
            },
        ),
    )
    registry = _registry_with_tags(base, (tag,))
    payload = _payload(inventory_a, registry=registry)

    present = _observe(
        raw_a,
        payload,
        mode=ObjectSceneTranscriptMode.REGISTERED_EVALUATION,
        registry=registry,
    )
    assert present.status is PrototypeSceneObserverStatus.SUCCESS
    assert present.transcript is not None
    cell = present.transcript.objects[0].registered_tag_cells[0]
    assert cell.disposition is Disposition.PRESENT
    assert tuple(item.witness_id for item in cell.witness_cells) == (
        "witness_00",
        "witness_01",
    )
    assert cell.tag_digest == tag.tag_digest

    absent_payload = deepcopy(payload)
    absent_payload["objects"][0]["registered_tags"][0]["witness_cells"][
        1
    ]["state"] = "absent"
    absent = _observe(
        raw_a,
        absent_payload,
        mode=ObjectSceneTranscriptMode.REGISTERED_EVALUATION,
        registry=registry,
    )
    assert absent.status is PrototypeSceneObserverStatus.SUCCESS
    assert absent.transcript.objects[0].registered_tag_cells[
        0
    ].disposition is Disposition.CERTIFIED_ABSENT

    indeterminate_payload = deepcopy(payload)
    indeterminate_payload["objects"][0]["registered_tags"][0][
        "witness_cells"
    ][0]["state"] = "indeterminate"
    indeterminate = _observe(
        raw_a,
        indeterminate_payload,
        mode=ObjectSceneTranscriptMode.REGISTERED_EVALUATION,
        registry=registry,
    )
    assert indeterminate.status is PrototypeSceneObserverStatus.SUCCESS
    assert indeterminate.transcript.objects[0].registered_tag_cells[
        0
    ].disposition is Disposition.INDETERMINATE

    reordered = deepcopy(payload)
    reordered["objects"][0]["registered_tags"][0][
        "witness_cells"
    ].reverse()
    rejected = _observe(
        raw_a,
        reordered,
        mode=ObjectSceneTranscriptMode.REGISTERED_EVALUATION,
        registry=registry,
    )
    assert rejected.status is PrototypeSceneObserverStatus.PARSER_ERROR
    assert rejected.transcript is None
    assert lookup_object_scene_soft_tag(
        rejected, "object_0000", "bird-like object"
    ).disposition is Disposition.ERROR


def test_resealed_registered_witness_tamper_fails_cold_payload_replay():
    raw_a, raw_b = _scene(0), _scene(2)
    inventory_a = extract_object_scene_proposal_inventory(raw_a)
    inventory_b = extract_object_scene_proposal_inventory(raw_b)
    first = _observe(raw_a, _payload(inventory_a))
    second = _observe(raw_b, _payload(inventory_b), context=CONTEXT_B)
    registry = freeze_object_scene_soft_tag_registry(
        (first.transcript, second.transcript)
    )
    artifact = _observe(
        raw_a,
        _payload(inventory_a, registry=registry),
        mode=ObjectSceneTranscriptMode.REGISTERED_EVALUATION,
        registry=registry,
    )
    tampered = deepcopy(artifact.to_data())
    row = tampered["transcript"]["objects"][0]
    tag_cell = row["registered_tag_cells"][0]
    witness_cell = tag_cell["witness_cells"][0]
    witness_cell["evidence"] = "tampered visible witness evidence"
    witness_cell["cell_digest"] = frontend.canonical_digest(
        {
            key: value
            for key, value in witness_cell.items()
            if key != "cell_digest"
        }
    )
    tag_cell["cell_digest"] = frontend.canonical_digest(
        {
            key: value
            for key, value in tag_cell.items()
            if key != "cell_digest"
        }
    )
    row["row_digest"] = frontend.canonical_digest(
        {key: value for key, value in row.items() if key != "row_digest"}
    )
    transcript = tampered["transcript"]
    transcript["transcript_digest"] = frontend.canonical_digest(
        {
            key: value
            for key, value in transcript.items()
            if key != "transcript_digest"
        }
    )
    tampered["artifact_digest"] = frontend.canonical_digest(
        {key: value for key, value in tampered.items() if key != "artifact_digest"}
    )
    resealed = ObjectSceneTranscriptArtifact.from_data(tampered)

    with pytest.raises(
        ObjectSceneVisualFrontendError,
        match="transcript replay differs",
    ):
        verify_object_scene_transcript_artifact(
            resealed,
            raw_a,
            expected_scene_id="opaque-scene",
            expected_observation_context_digest=CONTEXT_A,
            expected_panel_sha256=hashlib.sha256(raw_a).hexdigest(),
            expected_artifact_digest=resealed.artifact_digest,
        )


def test_direct_transcript_lookups_reject_resealed_card_binding_tamper():
    raw_a, raw_b = _scene(0), _scene(2)
    inventory_a = extract_object_scene_proposal_inventory(raw_a)
    inventory_b = extract_object_scene_proposal_inventory(raw_b)
    first = _observe(
        raw_a,
        _payload(
            inventory_a,
            open_tags=("bird-like object",),
            panel_open_tags=("separated forms",),
        ),
    )
    second = _observe(
        raw_b,
        _payload(
            inventory_b,
            open_tags=("bird-like object",),
            panel_open_tags=("separated forms",),
        ),
        context=CONTEXT_B,
    )
    registry = freeze_object_scene_soft_tag_registry(
        (first.transcript, second.transcript)
    )
    artifact = _observe(
        raw_a,
        _payload(inventory_a, registry=registry),
        mode=ObjectSceneTranscriptMode.REGISTERED_EVALUATION,
        registry=registry,
    )
    assert artifact.transcript is not None

    entity_raw = deepcopy(artifact.transcript.to_data())
    entity_row = entity_raw["objects"][0]
    entity_cell = entity_row["registered_tag_cells"][0]
    entity_cell["tag_digest"] = "f" * 64
    entity_cell["cell_digest"] = frontend.canonical_digest(
        {
            key: value
            for key, value in entity_cell.items()
            if key != "cell_digest"
        }
    )
    entity_row["row_digest"] = frontend.canonical_digest(
        {
            key: value
            for key, value in entity_row.items()
            if key != "row_digest"
        }
    )
    entity_raw["transcript_digest"] = frontend.canonical_digest(
        {
            key: value
            for key, value in entity_raw.items()
            if key != "transcript_digest"
        }
    )
    entity_transcript = frontend.ObjectSceneTranscript.from_data(entity_raw)
    with pytest.raises(
        ObjectSceneVisualFrontendError,
        match="registered cell card binding differs",
    ):
        lookup_object_scene_soft_tag(
            entity_transcript,
            "object_0000",
            "bird-like object",
            registry=registry,
        )

    panel_raw = deepcopy(artifact.transcript.to_data())
    panel_cell = panel_raw["panel_registered_tag_cells"][0]
    panel_witness = panel_cell["witness_cells"][0]
    panel_witness["witness_digest"] = "e" * 64
    panel_witness["cell_digest"] = frontend.canonical_digest(
        {
            key: value
            for key, value in panel_witness.items()
            if key != "cell_digest"
        }
    )
    panel_cell["cell_digest"] = frontend.canonical_digest(
        {
            key: value
            for key, value in panel_cell.items()
            if key != "cell_digest"
        }
    )
    panel_raw["transcript_digest"] = frontend.canonical_digest(
        {
            key: value
            for key, value in panel_raw.items()
            if key != "transcript_digest"
        }
    )
    panel_transcript = frontend.ObjectSceneTranscript.from_data(panel_raw)
    with pytest.raises(
        ObjectSceneVisualFrontendError,
        match="registered cell card binding differs",
    ):
        lookup_object_scene_panel_soft_tag(
            panel_transcript,
            "separated forms",
            registry=registry,
        )


def test_registered_passes_have_identical_visible_envelopes_and_distinct_contexts():
    raw_a, raw_b = _scene(0), _scene(2)
    inv_a, inv_b = extract_object_scene_proposal_inventory(raw_a), extract_object_scene_proposal_inventory(raw_b)
    first = _observe(raw_a, _payload(inv_a))
    second = _observe(raw_b, _payload(inv_b), context=CONTEXT_B)
    registry = freeze_object_scene_soft_tag_registry((first.transcript, second.transcript))
    registered_payload = _payload(inv_a, registry=registry)
    envelopes = []

    pass_a = _observe(raw_a, registered_payload, context=CONTEXT_A, mode=ObjectSceneTranscriptMode.REGISTERED_EVALUATION, registry=registry, envelopes=envelopes)
    pass_b = _observe(raw_a, registered_payload, context=CONTEXT_B, mode=ObjectSceneTranscriptMode.REGISTERED_EVALUATION, registry=registry, envelopes=envelopes)

    assert pass_a.status is pass_b.status is PrototypeSceneObserverStatus.SUCCESS
    assert envelopes[0] == envelopes[1]
    assert pass_a.observation_context_digest != pass_b.observation_context_digest
    assert pass_a.artifact_digest != pass_b.artifact_digest
    assert pass_a.transcript.objects[0].registered_tag_cells[0].support.to_data() == {"lower": 1, "upper": 1}
    assert lookup_object_scene_soft_tag(pass_a, "object_0000", "unregistered form").disposition is Disposition.INDETERMINATE


def test_registered_payload_and_schema_exhaust_the_scoped_partitions():
    raw_a, raw_b = _scene(0), _scene(2)
    inv_a = extract_object_scene_proposal_inventory(raw_a)
    inv_b = extract_object_scene_proposal_inventory(raw_b)
    discovery_a = _observe(
        raw_a,
        _payload(
            inv_a,
            open_tags=("bird-like object",),
            panel_open_tags=("separated forms",),
        ),
    )
    discovery_b = _observe(
        raw_b,
        _payload(
            inv_b,
            open_tags=("bird-like object",),
            panel_open_tags=("separated forms",),
        ),
        context=CONTEXT_B,
    )
    assert discovery_a.transcript is not None
    assert discovery_b.transcript is not None
    registry = freeze_object_scene_soft_tag_registry(
        (discovery_a.transcript, discovery_b.transcript)
    )
    assert tuple((item.tag_id, item.scope) for item in registry.tags) == (
        ("tag_0000", "entity"),
        ("tag_0001", "panel"),
    )

    schema = object_scene_transcript_output_schema(
        inv_a, ObjectSceneTranscriptMode.REGISTERED_EVALUATION, registry
    )
    properties = schema["properties"]
    panel_ids = properties["panel"]["properties"]["registered_tags"][
        "items"
    ]["properties"]["tag_id"]["enum"]
    entity_ids = properties["objects"]["items"]["properties"][
        "registered_tags"
    ]["items"]["properties"]["tag_id"]["enum"]
    assert panel_ids == ["tag_0001"]
    assert entity_ids == ["tag_0000"]

    artifact = _observe(
        raw_a,
        _payload(inv_a, registry=registry),
        mode=ObjectSceneTranscriptMode.REGISTERED_EVALUATION,
        registry=registry,
    )
    assert artifact.status is PrototypeSceneObserverStatus.SUCCESS
    assert artifact.transcript is not None
    transcript = artifact.transcript
    assert transcript.registered_panel_tag_ids == ("tag_0001",)
    assert transcript.registered_entity_tag_ids == ("tag_0000",)
    assert tuple(
        cell.tag_id for cell in transcript.panel_registered_tag_cells
    ) == ("tag_0001",)
    assert all(
        tuple(cell.tag_id for cell in row.registered_tag_cells)
        == ("tag_0000",)
        for row in transcript.objects
    )
    assert lookup_object_scene_panel_soft_tag(
        artifact, "separated forms"
    ).disposition is Disposition.PRESENT
    assert lookup_object_scene_soft_tag(
        artifact, "object_0000", "bird-like object"
    ).disposition is Disposition.PRESENT

    wrong_panel = _payload(inv_a, registry=registry)
    wrong_panel["panel"]["registered_tags"] = deepcopy(
        wrong_panel["objects"][0]["registered_tags"]
    )
    rejected = _observe(
        raw_a,
        wrong_panel,
        mode=ObjectSceneTranscriptMode.REGISTERED_EVALUATION,
        registry=registry,
    )
    assert rejected.status is PrototypeSceneObserverStatus.PARSER_ERROR
    assert lookup_object_scene_panel_soft_tag(
        rejected, "separated forms"
    ).disposition is Disposition.ERROR


def test_scope_partition_tampering_is_rejected_by_transcript_and_artifact():
    raw_a, raw_b = _scene(0), _scene(2)
    inv_a = extract_object_scene_proposal_inventory(raw_a)
    inv_b = extract_object_scene_proposal_inventory(raw_b)
    first = _observe(
        raw_a,
        _payload(
            inv_a,
            open_tags=("bird-like object",),
            panel_open_tags=("separated forms",),
        ),
    )
    second = _observe(
        raw_b,
        _payload(
            inv_b,
            open_tags=("bird-like object",),
            panel_open_tags=("separated forms",),
        ),
        context=CONTEXT_B,
    )
    assert first.transcript is not None and second.transcript is not None
    registry = freeze_object_scene_soft_tag_registry(
        (first.transcript, second.transcript)
    )
    artifact = _observe(
        raw_a,
        _payload(inv_a, registry=registry),
        mode=ObjectSceneTranscriptMode.REGISTERED_EVALUATION,
        registry=registry,
    )
    assert artifact.transcript is not None
    transcript = artifact.transcript

    with pytest.raises(ObjectSceneVisualFrontendError):
        replace(
            transcript,
            registered_panel_tag_ids=transcript.registered_entity_tag_ids,
        )

    rows = []
    for row in transcript.objects:
        row_values = {
            "object_id": row.object_id,
            "crop_receipt_digest": row.crop_receipt_digest,
            "summary": row.summary,
            "count_cells": row.count_cells,
            "qualitative_cells": row.qualitative_cells,
            "open_tags": row.open_tags,
            "registered_tag_cells": transcript.panel_registered_tag_cells,
        }
        provisional_row = object.__new__(type(row))
        for name, value in row_values.items():
            object.__setattr__(provisional_row, name, value)
        rows.append(
            type(row)(
                **row_values,
                row_digest=frontend.canonical_digest(
                    frontend._row_content(provisional_row)
                ),
            )
        )
    transcript_values = {
        "panel_digest": transcript.panel_digest,
        "inventory_digest": transcript.inventory_digest,
        "mode": transcript.mode,
        "registry_digest": transcript.registry_digest,
        "panel_summary": transcript.panel_summary,
        "panel_open_tags": transcript.panel_open_tags,
        "panel_registered_tag_cells": transcript.objects[
            0
        ].registered_tag_cells,
        "dropped_open_tags": transcript.dropped_open_tags,
        "registered_panel_tag_ids": transcript.registered_entity_tag_ids,
        "registered_entity_tag_ids": transcript.registered_panel_tag_ids,
        "objects": tuple(rows),
    }
    provisional_transcript = object.__new__(type(transcript))
    for name, value in transcript_values.items():
        object.__setattr__(provisional_transcript, name, value)
    structurally_resealed = type(transcript)(
        **transcript_values,
        transcript_digest=frontend.canonical_digest(
            frontend._transcript_content(provisional_transcript)
        ),
    )
    with pytest.raises(
        ObjectSceneVisualFrontendError,
        match="registry scope partition",
    ):
        replace(artifact, transcript=structurally_resealed)


def test_zero_proposal_panel_transcripts_replay_in_both_modes():
    raw_a, raw_b = _blank_scene(96), _blank_scene(97)
    inv_a = extract_object_scene_proposal_inventory(raw_a)
    inv_b = extract_object_scene_proposal_inventory(raw_b)
    assert inv_a.objects == inv_b.objects == ()

    payload_a = _payload(
        inv_a, open_tags=(), panel_open_tags=("uniform white field",)
    )
    payload_a["panel"]["summary"] = "uniform white field fills the panel"
    calls = []
    first = _observe(raw_a, payload_a, calls=calls)
    payload_b = _payload(
        inv_b, open_tags=(), panel_open_tags=("uniform white field",)
    )
    payload_b["panel"]["summary"] = "uniform white field fills the panel"
    second = _observe(raw_b, payload_b, context=CONTEXT_B)
    assert first.status is second.status is PrototypeSceneObserverStatus.SUCCESS
    assert calls[0][0] == ("panel.png",)
    assert first.transcript is not None and second.transcript is not None
    assert first.transcript.objects == ()
    assert lookup_object_scene_panel_soft_tag(
        first, "uniform white field"
    ).disposition is Disposition.PRESENT

    registry = freeze_object_scene_soft_tag_registry(
        (first.transcript, second.transcript)
    )
    assert tuple((item.scope, item.tag) for item in registry.tags) == (
        ("panel", "uniform white field"),
    )
    registered_payload = _payload(inv_a, open_tags=(), registry=registry)
    registered_payload["panel"]["summary"] = (
        "uniform white field fills the panel"
    )
    registered = _observe(
        raw_a,
        registered_payload,
        mode=ObjectSceneTranscriptMode.REGISTERED_EVALUATION,
        registry=registry,
    )
    assert registered.status is PrototypeSceneObserverStatus.SUCCESS
    assert registered.transcript is not None
    assert registered.transcript.registered_panel_tag_ids == ("tag_0000",)
    assert registered.transcript.registered_entity_tag_ids == ()
    assert registered.transcript.objects == ()
    assert verify_object_scene_transcript_artifact(
        registered,
        raw_a,
        expected_scene_id="opaque-scene",
        expected_observation_context_digest=CONTEXT_A,
        expected_panel_sha256=hashlib.sha256(raw_a).hexdigest(),
        expected_artifact_digest=registered.artifact_digest,
    ) == registered


def test_missing_registered_cell_is_parser_error_never_absence():
    raw_a, raw_b = _scene(0), _scene(2)
    inv_a, inv_b = extract_object_scene_proposal_inventory(raw_a), extract_object_scene_proposal_inventory(raw_b)
    first = _observe(raw_a, _payload(inv_a))
    second = _observe(raw_b, _payload(inv_b), context=CONTEXT_B)
    registry = freeze_object_scene_soft_tag_registry((first.transcript, second.transcript))
    malformed = _payload(inv_a, registry=registry)
    malformed["objects"][0]["registered_tags"] = []

    artifact = _observe(raw_a, malformed, mode=ObjectSceneTranscriptMode.REGISTERED_EVALUATION, registry=registry)
    assert artifact.status is PrototypeSceneObserverStatus.PARSER_ERROR
    assert artifact.transcript is None
    assert lookup_object_scene_soft_tag(artifact, "object_0000", "bird-like object").disposition is Disposition.ERROR
    verify_object_scene_transcript_artifact(
        artifact,
        raw_a,
        expected_scene_id="opaque-scene",
        expected_observation_context_digest=CONTEXT_A,
        expected_panel_sha256=hashlib.sha256(raw_a).hexdigest(),
        expected_artifact_digest=artifact.artifact_digest,
    )


def test_invalid_registered_witness_value_is_local_error_not_panel_error():
    raw_a, raw_b = _scene(0), _scene(2)
    inv_a = extract_object_scene_proposal_inventory(raw_a)
    inv_b = extract_object_scene_proposal_inventory(raw_b)
    first = _observe(raw_a, _payload(inv_a))
    second = _observe(raw_b, _payload(inv_b), context=CONTEXT_B)
    registry = freeze_object_scene_soft_tag_registry(
        (first.transcript, second.transcript)
    )
    payload = _payload(inv_a, registry=registry)
    rejected_evidence = (
        "the candidate widened portions do not form a clear opposing pair"
    )
    payload["objects"][0]["registered_tags"][0]["witness_cells"][0][
        "evidence"
    ] = rejected_evidence

    artifact = _observe(
        raw_a,
        payload,
        mode=ObjectSceneTranscriptMode.REGISTERED_EVALUATION,
        registry=registry,
    )

    assert artifact.status is PrototypeSceneObserverStatus.SUCCESS
    assert artifact.transcript is not None
    assert artifact.model_payload["objects"][0]["registered_tags"][0][
        "witness_cells"
    ][0]["evidence"] == rejected_evidence
    assert lookup_object_scene_soft_tag(
        artifact, "object_0000", "bird-like object"
    ).disposition is Disposition.ERROR
    assert lookup_object_scene_soft_tag(
        artifact, "object_0001", "bird-like object"
    ).disposition is Disposition.PRESENT
    verify_object_scene_transcript_artifact(
        artifact,
        raw_a,
        expected_scene_id="opaque-scene",
        expected_observation_context_digest=CONTEXT_A,
        expected_panel_sha256=hashlib.sha256(raw_a).hexdigest(),
        expected_artifact_digest=artifact.artifact_digest,
    )


def test_tampering_and_lean_import_are_rejected():
    raw = _scene()
    inventory = extract_object_scene_proposal_inventory(raw)
    artifact = _observe(raw, _payload(inventory))
    tampered = deepcopy(artifact.to_data())
    tampered["physical_call_count"] = 2
    with pytest.raises(ObjectSceneVisualFrontendError):
        ObjectSceneTranscriptArtifact.from_data(tampered)

    tree = ast.parse(Path(frontend.__file__).read_text())
    imported = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert "lean" not in {name.lower() for name in imported}

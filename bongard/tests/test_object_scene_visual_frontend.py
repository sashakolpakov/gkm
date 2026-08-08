from __future__ import annotations

import ast
from copy import deepcopy
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
    OBJECT_SCENE_QUALITATIVE_OBSERVABLE_IDS,
    ObjectSceneProposalInventory,
    ObjectSceneSoftTagRegistry,
    ObjectSceneTranscriptArtifact,
    ObjectSceneTranscriptMode,
    ObjectSceneVisualFrontendError,
    extract_object_scene_proposal_inventory,
    freeze_object_scene_soft_tag_registry,
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


def _payload(inventory, *, open_tags=("bird-like object",), registry=None):
    rows = []
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
                    {
                        "tag_id": tag.tag_id,
                        "state": "present" if tag.tag == "bird-like object" else "indeterminate",
                        "evidence": "visible silhouette supports phrase" if tag.tag == "bird-like object" else "visible detail is unresolved",
                    }
                    for tag in registry.tags
                ],
            }
        )
    return {"objects": rows}


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


def test_registry_uses_distinct_panel_frequency_and_persists_every_drop():
    raw_a, raw_b = _scene(0), _scene(2)
    inventory_a = extract_object_scene_proposal_inventory(raw_a)
    inventory_b = extract_object_scene_proposal_inventory(raw_b)
    discovery_a = _observe(raw_a, _payload(inventory_a, open_tags=("bird-like object", "pointed form")))
    discovery_b = _observe(raw_b, _payload(inventory_b, open_tags=("bird-like object", "sector form")), context=CONTEXT_B)
    assert discovery_a.transcript is not None and discovery_b.transcript is not None

    registry = freeze_object_scene_soft_tag_registry((discovery_a.transcript, discovery_b.transcript))
    assert tuple((item.tag_id, item.tag, item.distinct_panel_count) for item in registry.tags) == (("tag_0000", "bird-like object", 2),)
    assert tuple((item.tag, item.reason) for item in registry.dropped_tags) == (
        ("pointed form", "seen_on_fewer_than_2_panels"),
        ("sector form", "seen_on_fewer_than_2_panels"),
    )
    assert ObjectSceneSoftTagRegistry.from_data(registry.to_data()) == registry
    assert verify_object_scene_soft_tag_registry(
        registry,
        (discovery_a.transcript, discovery_b.transcript),
        expected_registry_digest=registry.registry_digest,
    ) == registry


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

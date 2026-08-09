from __future__ import annotations

from copy import deepcopy
from io import BytesIO
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from bongard.canonical import canonical_digest
from bongard.object_scene_anchor_atlas import (
    OBJECT_SCENE_ANCHOR_ATLAS_HEIGHT_PIXELS,
    OBJECT_SCENE_ANCHOR_ATLAS_MAX_SLOTS,
    OBJECT_SCENE_ANCHOR_ATLAS_WIDTH_PIXELS,
    ObjectSceneAnchorAtlas,
    ObjectSceneAnchorAtlasError,
    render_object_scene_anchor_atlas,
    verify_object_scene_anchor_atlas,
)
from bongard.object_scene_anchor_catalog import _make_entry
from bongard.object_scene_anchor_salience import (
    AnchorSalienceLimits,
    extract_object_scene_anchor_salience,
)
from bongard.object_scene_visual_frontend import (
    OBJECT_SCENE_CANONICAL_SCENARIO_ID,
)


def _mask() -> np.ndarray:
    """One plus junction and one isolated compact component."""

    mask = np.zeros((50, 50), dtype=np.bool_)
    mask[30, 12:35] = True
    mask[19:42, 23] = True
    mask[3, 46] = True
    return mask


def _entry(limits: AnchorSalienceLimits | None = None):
    mask = _mask()
    salience = extract_object_scene_anchor_salience(
        mask, "object_0000", limits
    )
    receipt = SimpleNamespace(
        object_id="object_0000",
        receipt_digest="1" * 64,
        lineage_id="lineage-00000000",
        lineage_digest="2" * 64,
        scenario_id=OBJECT_SCENE_CANONICAL_SCENARIO_ID,
        hypothesis_id="hypothesis-00000000",
        hypothesis_digest="3" * 64,
        masked_crop_pixel_digest="4" * 64,
    )
    return _make_entry(
        inventory_index=0,
        receipt=receipt,
        mask=mask,
        salience=salience,
    )


def _all_keys(value: object):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _all_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _all_keys(item)


def _reseal(value: dict[str, object], digest_key: str) -> None:
    value[digest_key] = canonical_digest(
        {key: item for key, item in value.items() if key != digest_key}
    )


def _reseal_slot_map(value: dict[str, object]) -> None:
    value["slot_map_digest"] = canonical_digest(
        {
            "schema": "gkm.object-scene-anchor-atlas-slot-map.v1",
            "order": "row-major-exhaustive-prefix",
            "slots": value["slots"],
        }
    )
    _reseal(value, "artifact_digest")


@pytest.fixture(scope="module")
def clean_render():
    entry = _entry()
    manifest = entry.decision_manifest
    artifact, png = render_object_scene_anchor_atlas(manifest)
    assert png is not None
    return manifest, artifact, png


def test_exact_roundtrip_replay_and_grayscale_png(clean_render) -> None:
    manifest, artifact, png = clean_render
    restored = ObjectSceneAnchorAtlas.from_data(artifact.to_data())
    second, second_png = render_object_scene_anchor_atlas(manifest)

    assert restored == artifact == second
    assert png == second_png
    assert verify_object_scene_anchor_atlas(artifact, png, manifest) == artifact
    with Image.open(BytesIO(png)) as image:
        assert image.mode == "L"
        assert image.size == (
            OBJECT_SCENE_ANCHOR_ATLAS_WIDTH_PIXELS,
            OBJECT_SCENE_ANCHOR_ATLAS_HEIGHT_PIXELS,
        )
        assert image.getextrema()[0] < image.getextrema()[1]


def test_slot_map_is_complete_ordered_and_never_truncated(clean_render) -> None:
    manifest, artifact, _ = clean_render
    graph = manifest.selected_graph
    assert graph is not None

    expected_subjects = (
        manifest.object_id,
        *manifest.selected_anchor_ids,
        *manifest.selected_frame_ids,
    )
    assert OBJECT_SCENE_ANCHOR_ATLAS_MAX_SLOTS == 17
    assert artifact.slot_count == len(expected_subjects)
    assert artifact.slot_count == (
        1
        + len(graph.parts)
        + len(graph.compact_components)
        + len(graph.cyclic_frames)
    )
    assert tuple(slot.subject_id for slot in artifact.slots) == expected_subjects
    assert artifact.slots[0].slot_kind == "whole_entity"
    assert set(artifact.slots[0].highlight_part_ids) == {
        item.part_id for item in graph.parts
    }
    assert set(artifact.slots[0].highlight_compact_ids) == {
        item.compact_id for item in graph.compact_components
    }

    frame_by_id = {item.frame_id: item for item in graph.cyclic_frames}
    for slot in artifact.slots:
        if slot.slot_kind != "cyclic_frame":
            continue
        frame = frame_by_id[slot.subject_id]
        assert slot.highlight_join_ids == (frame.join_id,)
        assert slot.highlight_part_ids == frame.clockwise_incident_part_ids
        assert (
            slot.highlight_tangent_points_q16
            == frame.clockwise_tangent_points_q16
        )


def test_atlas_identity_is_python_only_and_excludes_raw_audit_inputs(
    clean_render,
) -> None:
    _, artifact, _ = clean_render
    data = artifact.to_data()

    assert not any("lean" in key.lower() for key in _all_keys(data))
    assert data["python_is_canonical_authority"] is True
    assert data["selected_decision_manifest_is_only_geometry_input"] is True
    assert data["raw_graph_consumed"] is False
    assert data["audit_graph_consumed"] is False
    assert data["fresh_or_query_pixels_consumed"] is False
    assert data["top_k_or_truncation_applied"] is False
    assert "source_entry_digest" not in data
    assert "source_salience_artifact_digest" not in data
    assert "entry_digest" not in data
    assert "salience_artifact_digest" not in data
    assert "raw_graph" not in data
    assert "audit_graph" not in data


def test_tampered_metadata_or_png_fails_exact_replay(clean_render) -> None:
    manifest, artifact, png = clean_render
    data = deepcopy(artifact.to_data())
    data["png_digest"] = "0" * 64
    _reseal(data, "artifact_digest")
    tampered = ObjectSceneAnchorAtlas.from_data(data)

    with pytest.raises(ObjectSceneAnchorAtlasError, match="exact selected-manifest"):
        verify_object_scene_anchor_atlas(tampered, png, manifest)

    changed_png = png[:-1] + bytes((png[-1] ^ 1,))
    with pytest.raises(ObjectSceneAnchorAtlasError, match="exact selected-manifest"):
        verify_object_scene_anchor_atlas(artifact, changed_png, manifest)


def test_standalone_roundtrip_rejects_resealed_truncation_or_forged_subject(
    clean_render,
) -> None:
    _, artifact, _ = clean_render

    truncated = deepcopy(artifact.to_data())
    truncated["slots"].pop()
    truncated["slot_count"] = len(truncated["slots"])
    _reseal_slot_map(truncated)
    with pytest.raises(
        ObjectSceneAnchorAtlasError,
        match="embedded selected decision graph",
    ):
        ObjectSceneAnchorAtlas.from_data(truncated)

    forged = deepcopy(artifact.to_data())
    forged_slot = forged["slots"][1]
    forged_slot["subject_digest"] = "0" * 64
    _reseal(forged_slot, "slot_digest")
    _reseal_slot_map(forged)
    with pytest.raises(
        ObjectSceneAnchorAtlasError,
        match="embedded selected decision graph",
    ):
        ObjectSceneAnchorAtlas.from_data(forged)


def test_nonclean_salience_returns_typed_gap_without_partial_atlas() -> None:
    entry = _entry(AnchorSalienceLimits(max_padded_pixels=1))
    manifest = entry.decision_manifest
    artifact, png = render_object_scene_anchor_atlas(manifest)

    assert manifest.salience_state == "indeterminate"
    assert artifact.status.state == "indeterminate"
    assert artifact.status.reason == manifest.salience_reason
    assert artifact.selected_graph_artifact_digest is None
    assert artifact.slot_count == 0
    assert artifact.slots == ()
    assert artifact.image_width_pixels == artifact.image_height_pixels == 0
    assert artifact.png_byte_count == 0
    assert artifact.png_digest is None
    assert png is None
    assert verify_object_scene_anchor_atlas(artifact, None, manifest) == artifact

    with pytest.raises(ObjectSceneAnchorAtlasError, match="exact selected-manifest"):
        verify_object_scene_anchor_atlas(artifact, b"", manifest)

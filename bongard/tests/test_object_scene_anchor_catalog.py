from __future__ import annotations

from copy import deepcopy
from io import BytesIO

from PIL import Image, ImageDraw
import numpy as np
import pytest

from bongard.canonical import canonical_digest
from bongard.object_scene_anchor_catalog import (
    ObjectSceneAnchorCatalog,
    ObjectSceneAnchorCatalogError,
    extract_object_scene_anchor_catalog,
    object_scene_proposal_bool_mask_digest,
    verify_object_scene_anchor_catalog,
)
from bongard.object_scene_anchor_salience import (
    AnchorSalienceLimits,
    verify_object_scene_anchor_salience,
)
from bongard import object_scene_visual_frontend as frontend
from bongard import prototype_object_hypotheses as hypotheses_module
from bongard import visual_witnesses
from bongard.object_scene_visual_frontend import (
    OBJECT_SCENE_CANONICAL_SCENARIO_ID,
    extract_object_scene_proposal_inventory,
)
from bongard.prototype_object_hypotheses import extract_object_hypothesis_packet


def _scene(shift: int = 0) -> bytes:
    image = Image.new("RGB", (64, 48), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((5 + shift, 8, 18 + shift, 25), fill="black")
    draw.ellipse((40, 10, 53, 24), fill="black")
    output = BytesIO()
    image.save(output, format="PNG", optimize=False)
    return output.getvalue()


def _morphology_scene() -> bytes:
    image = Image.new("RGB", (48, 48), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((10, 10, 32, 32), fill="black")
    draw.point((21, 21), fill="white")
    output = BytesIO()
    image.save(output, format="PNG", optimize=False)
    return output.getvalue()


@pytest.fixture(scope="module")
def frozen_catalog():
    raw = _scene()
    inventory = extract_object_scene_proposal_inventory(raw)
    catalog = extract_object_scene_anchor_catalog(raw, inventory)
    return raw, inventory, catalog


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


def _canonical_masks(raw: bytes, packet):
    visual = visual_witnesses.extract_visual_witnesses(raw)
    strength = visual_witnesses._decode_png(raw)
    scenario = next(
        item
        for item in visual.scenarios
        if item.scenario_id == OBJECT_SCENE_CANONICAL_SCENARIO_ID
    )
    component_masks = hypotheses_module._component_masks(strength, scenario)
    by_component = {
        component.component_id: mask
        for component, mask in zip(scenario.components, component_masks, strict=True)
    }
    result = {}
    for hypothesis in next(
        item
        for item in packet.scenarios
        if item.scenario_id == OBJECT_SCENE_CANONICAL_SCENARIO_ID
    ).hypotheses:
        union = np.zeros_like(strength, dtype=np.bool_)
        for component_id in hypothesis.source_component_ids:
            union |= by_component[component_id]
        x0, y0, x1, y1 = visual_witnesses._bbox(union)
        result[(hypothesis.scenario_id, hypothesis.hypothesis_id)] = (
            np.ascontiguousarray(union[y0:y1, x0:x1], dtype=np.bool_)
        )
    return result


def test_catalog_roundtrip_exhausts_inventory_and_freezes_exact_bool_masks(
    frozen_catalog,
) -> None:
    raw, inventory, catalog = frozen_catalog
    restored = ObjectSceneAnchorCatalog.from_data(catalog.to_data())

    assert restored == catalog
    assert catalog.proposal_count == len(inventory.objects)
    assert catalog.object_ids == tuple(item.object_id for item in inventory.objects)
    assert tuple(catalog.by_object_id) == catalog.object_ids
    assert tuple(catalog.to_data()["objects"]) == catalog.object_ids

    hypotheses = extract_object_hypothesis_packet(raw)
    crops = frontend._hypothesis_crop_map(raw, hypotheses)
    masks = _canonical_masks(raw, hypotheses)
    for receipt, entry in zip(inventory.objects, catalog.entries, strict=True):
        crop = crops[(receipt.scenario_id, receipt.hypothesis_id)]
        mask = masks[(receipt.scenario_id, receipt.hypothesis_id)]
        assert entry.crop_receipt_digest == receipt.receipt_digest
        assert entry.masked_crop_pixel_digest == receipt.masked_crop_pixel_digest
        assert entry.bool_mask_digest == object_scene_proposal_bool_mask_digest(mask)
        assert entry.bool_mask_digest == entry.salience.source_mask_digest
        assert entry.foreground_pixel_count == receipt.union_area_pixels
        assert (
            verify_object_scene_anchor_salience(
                entry.salience,
                expected_mask=mask,
                expected_object_id=receipt.object_id,
            )
            == entry.salience
        )


def test_decision_manifest_contains_only_selected_graph_and_python_keys(
    frozen_catalog,
) -> None:
    _, _, catalog = frozen_catalog
    data = catalog.to_data()

    assert not any("lean" in key.lower() for key in _all_keys(data))
    assert data["python_is_canonical_authority"] is True
    assert data["raw_graph_decision_bearing"] is False
    assert data["audit_graph_decision_bearing"] is False
    for entry in catalog.entries:
        manifest = entry.decision_manifest
        manifest_data = manifest.to_data()
        assert "salience_artifact_digest" not in manifest_data
        assert "selected_support_counts" not in manifest_data
        assert "selected_attempt_index" not in manifest_data
        assert "selected_radius_pixels" not in manifest_data
        assert "raw_graph" not in manifest_data
        assert "audit_graph" not in manifest_data
        assert manifest_data["raw_graph_decision_bearing"] is False
        assert manifest_data["audit_graph_decision_bearing"] is False
        if entry.salience.status.state == "clean":
            assert manifest.decision_kind == "selected_graph"
            assert manifest.selected_graph == entry.salience.selected_graph
        else:
            assert manifest.decision_kind == "typed_salience_gap"
            assert manifest.selected_graph is None


def test_morphological_union_includes_zero_strength_pixels() -> None:
    raw = _morphology_scene()
    inventory = extract_object_scene_proposal_inventory(raw)
    catalog = extract_object_scene_anchor_catalog(raw, inventory)
    hypotheses = extract_object_hypothesis_packet(raw)
    crops = frontend._hypothesis_crop_map(raw, hypotheses)
    masks = _canonical_masks(raw, hypotheses)

    receipt = inventory.objects[0]
    key = (receipt.scenario_id, receipt.hypothesis_id)
    mask = masks[key]
    assert int(np.count_nonzero(crops[key])) < int(np.count_nonzero(mask))
    assert catalog.entries[0].foreground_pixel_count == receipt.union_area_pixels
    assert catalog.entries[0].bool_mask_digest == object_scene_proposal_bool_mask_digest(mask)


def test_nonclean_salience_is_a_typed_gap_without_partial_graphs() -> None:
    raw = _scene()
    inventory = extract_object_scene_proposal_inventory(raw)
    catalog = extract_object_scene_anchor_catalog(
        raw,
        inventory,
        AnchorSalienceLimits(max_padded_pixels=64),
    )

    assert catalog.proposal_count > 0
    for entry in catalog.entries:
        assert entry.salience.status.state == "indeterminate"
        manifest = entry.decision_manifest
        assert manifest.salience_state == "indeterminate"
        assert manifest.decision_kind == "typed_salience_gap"
        assert manifest.selected_graph is None
        assert manifest.selected_anchor_ids == ()
        assert manifest.selected_frame_ids == ()
        assert "selected_support_counts" not in manifest.to_data()
        assert "raw_graph" not in manifest.to_data()
        assert "audit_graph" not in manifest.to_data()


def test_resealed_attempt_to_make_raw_graph_decision_bearing_is_rejected(
    frozen_catalog,
) -> None:
    _, _, catalog = frozen_catalog
    data = deepcopy(catalog.to_data())
    object_id = catalog.object_ids[0]
    entry = data["objects"][object_id]
    manifest = entry["decision_manifest"]
    manifest["raw_graph_decision_bearing"] = True
    _reseal(manifest, "manifest_digest")
    _reseal(entry, "entry_digest")
    _reseal(data, "catalog_digest")

    with pytest.raises(ObjectSceneAnchorCatalogError, match="policy differs"):
        ObjectSceneAnchorCatalog.from_data(data)


def test_resealed_unbound_bool_mask_digest_is_rejected(frozen_catalog) -> None:
    _, _, catalog = frozen_catalog
    data = deepcopy(catalog.to_data())
    object_id = catalog.object_ids[0]
    entry = data["objects"][object_id]
    entry["bool_mask_digest"] = "0" * 64
    _reseal(entry, "entry_digest")
    _reseal(data, "catalog_digest")

    with pytest.raises(ObjectSceneAnchorCatalogError, match="salience binding"):
        ObjectSceneAnchorCatalog.from_data(data)


def test_verifier_rejects_wrong_png_and_wrong_inventory(frozen_catalog) -> None:
    raw, inventory, catalog = frozen_catalog
    assert verify_object_scene_anchor_catalog(catalog, raw, inventory) == catalog

    wrong_raw = _scene(shift=1)
    with pytest.raises(ValueError, match="exact PNG replay"):
        verify_object_scene_anchor_catalog(catalog, wrong_raw, inventory)

    wrong_inventory = extract_object_scene_proposal_inventory(wrong_raw)
    with pytest.raises(ValueError, match="exact PNG replay"):
        verify_object_scene_anchor_catalog(catalog, raw, wrong_inventory)

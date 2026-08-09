from __future__ import annotations

from copy import deepcopy
from io import BytesIO

from PIL import Image, ImageDraw
import pytest

from bongard.canonical import canonical_digest
from bongard.object_scene_anchor_catalog import extract_object_scene_anchor_catalog
from bongard.object_scene_anchor_panel_manifest import (
    ObjectSceneAnchorPanelDecisionManifest,
    ObjectSceneAnchorPanelManifestError,
    build_object_scene_anchor_panel_decision_manifest,
    verify_object_scene_anchor_panel_decision_manifest,
)
from bongard.object_scene_visual_frontend import (
    extract_object_scene_proposal_inventory,
)


def _scene(*, shift: int = 0) -> bytes:
    image = Image.new("RGB", (72, 48), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((4 + shift, 8, 18 + shift, 25), fill="black")
    draw.ellipse((43, 11, 58, 27), fill="black")
    output = BytesIO()
    image.save(output, format="PNG", optimize=False)
    return output.getvalue()


@pytest.fixture(scope="module")
def frozen_panel():
    png = _scene()
    inventory = extract_object_scene_proposal_inventory(png)
    catalog = extract_object_scene_anchor_catalog(png, inventory)
    manifest = build_object_scene_anchor_panel_decision_manifest(
        catalog, png, inventory
    )
    return png, inventory, catalog, manifest


def _all_keys(value: object):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _all_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _all_keys(item)


def _reseal(value: dict[str, object]) -> None:
    value["manifest_digest"] = canonical_digest(
        {key: item for key, item in value.items() if key != "manifest_digest"}
    )


def test_panel_manifest_is_complete_decision_only_and_round_trips(
    frozen_panel,
) -> None:
    _, inventory, catalog, manifest = frozen_panel
    data = manifest.to_data()

    assert ObjectSceneAnchorPanelDecisionManifest.from_data(data) == manifest
    assert manifest.proposal_count == len(inventory.objects) == 2
    assert manifest.object_ids == catalog.object_ids
    assert tuple(manifest.by_object_id) == catalog.object_ids
    assert tuple(item.object_id for item in manifest.object_decisions) == (
        catalog.object_ids
    )
    keys = tuple(_all_keys(data))
    assert all("entry_digest" not in key for key in keys)
    assert all("salience_artifact" not in key for key in keys)
    assert {
        key for key in keys if "raw_graph" in key
    } == {"raw_graph_decision_bearing"}
    assert {
        key for key in keys if "audit_graph" in key
    } == {"audit_graph_decision_bearing"}
    assert all("lean" not in key.casefold() for key in keys)
    assert data["complete_object_inventory_required"] is True
    assert data["object_omission_allowed"] is False


def test_resealed_object_omission_fails_pixel_bound_cold_replay(
    frozen_panel,
) -> None:
    png, inventory, catalog, manifest = frozen_panel
    data = deepcopy(manifest.to_data())
    data["proposal_count"] = 1
    data["object_ids"] = data["object_ids"][:1]
    data["object_decisions"] = data["object_decisions"][:1]
    _reseal(data)
    truncated = ObjectSceneAnchorPanelDecisionManifest.from_data(data)

    with pytest.raises(ObjectSceneAnchorPanelManifestError, match="exact catalog"):
        verify_object_scene_anchor_panel_decision_manifest(
            truncated, catalog, png, inventory
        )


def test_panel_manifest_rejects_reordering_and_wrong_policy(frozen_panel) -> None:
    _, _, _, manifest = frozen_panel
    reordered = deepcopy(manifest.to_data())
    reordered["object_decisions"].reverse()
    _reseal(reordered)
    with pytest.raises(ObjectSceneAnchorPanelManifestError, match="inventory"):
        ObjectSceneAnchorPanelDecisionManifest.from_data(reordered)

    wrong_policy = deepcopy(manifest.to_data())
    wrong_policy["object_omission_allowed"] = True
    _reseal(wrong_policy)
    with pytest.raises(ObjectSceneAnchorPanelManifestError, match="policy"):
        ObjectSceneAnchorPanelDecisionManifest.from_data(wrong_policy)


def test_panel_manifest_cold_verifier_rejects_wrong_pixels(frozen_panel) -> None:
    png, inventory, catalog, manifest = frozen_panel
    assert (
        verify_object_scene_anchor_panel_decision_manifest(
            manifest, catalog, png, inventory
        )
        == manifest
    )
    wrong_png = _scene(shift=2)
    with pytest.raises(ValueError, match="exact PNG replay"):
        verify_object_scene_anchor_panel_decision_manifest(
            manifest, catalog, wrong_png, inventory
        )

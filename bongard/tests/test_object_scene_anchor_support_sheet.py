from __future__ import annotations

from copy import deepcopy
import hashlib
from io import BytesIO

import numpy as np
from PIL import Image, ImageDraw
import pytest

from bongard import visual_witnesses as _visual
from bongard.canonical import canonical_digest
from bongard.object_scene_anchor_atlas import (
    object_scene_anchor_grayscale_png_byte_count,
    render_object_scene_anchor_atlas,
)
from bongard.object_scene_anchor_catalog import extract_object_scene_anchor_catalog
from bongard.object_scene_anchor_crop import render_object_scene_anchor_object_crop
from bongard.object_scene_anchor_panel_manifest import (
    build_object_scene_anchor_panel_decision_manifest,
)
from bongard.object_scene_anchor_salience import AnchorSalienceLimits
from bongard.object_scene_anchor_support_sheet import (
    OBJECT_SCENE_ANCHOR_SUPPORT_SHEET_MAX_PIXEL_COUNT,
    ObjectSceneAnchorSupportSheet,
    ObjectSceneAnchorSupportSheetError,
    ObjectSceneAnchorSupportSheetPlan,
    _layout,
    build_object_scene_anchor_support_sheet,
    plan_object_scene_anchor_support_sheet,
    verify_object_scene_anchor_support_sheet,
    verify_object_scene_anchor_support_sheet_plan,
)
from bongard.object_scene_visual_frontend import (
    extract_object_scene_proposal_inventory,
)


def _scene() -> bytes:
    image = Image.new("RGB", (96, 64), "white")
    draw = ImageDraw.Draw(image)
    draw.line((8, 30, 20, 8, 32, 30, 8, 30), fill="black", width=3)
    draw.line((58, 42, 70, 18, 84, 42, 58, 42), fill="black", width=3)
    output = BytesIO()
    image.save(output, format="PNG", optimize=False)
    return output.getvalue()


@pytest.fixture(scope="module")
def frozen_sheet():
    panel = _scene()
    inventory = extract_object_scene_proposal_inventory(panel)
    catalog = extract_object_scene_anchor_catalog(panel, inventory)
    manifest = build_object_scene_anchor_panel_decision_manifest(
        catalog, panel, inventory
    )
    artifact, sheet_png = build_object_scene_anchor_support_sheet(
        panel, inventory, catalog, manifest
    )
    return panel, inventory, catalog, manifest, artifact, sheet_png


def _png_pixels(payload: bytes) -> np.ndarray:
    with Image.open(BytesIO(payload)) as image:
        assert image.format == "PNG"
        assert image.mode == "L"
        return np.ascontiguousarray(np.asarray(image, dtype=np.uint8))


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


def test_support_sheet_is_deterministic_complete_and_cold_replayable(
    frozen_sheet,
) -> None:
    panel, inventory, catalog, manifest, artifact, sheet_png = frozen_sheet
    second, second_png = build_object_scene_anchor_support_sheet(
        panel, inventory, catalog, manifest
    )

    assert artifact == second
    assert sheet_png == second_png
    assert ObjectSceneAnchorSupportSheet.from_data(artifact.to_data()) == artifact
    assert (
        verify_object_scene_anchor_support_sheet(
            artifact,
            sheet_png,
            panel,
            inventory,
            catalog,
            manifest,
        )
        == artifact
    )
    assert artifact.panel_manifest_digest == manifest.manifest_digest
    assert artifact.proposal_count == manifest.proposal_count == 2
    assert artifact.object_ids == manifest.object_ids
    assert tuple(artifact.by_object_id) == manifest.object_ids


def test_support_sheet_plan_is_exact_canonical_and_matches_render(
    frozen_sheet,
) -> None:
    _, inventory, _, _, artifact, sheet_png = frozen_sheet
    plan = plan_object_scene_anchor_support_sheet(inventory)

    assert ObjectSceneAnchorSupportSheetPlan.from_data(plan.to_data()) == plan
    assert (
        verify_object_scene_anchor_support_sheet_plan(
            plan,
            inventory,
            expected_plan_digest=plan.plan_digest,
        )
        == plan
    )
    assert (plan.sheet_width_pixels, plan.sheet_height_pixels) == (
        artifact.sheet_width_pixels,
        artifact.sheet_height_pixels,
    )
    assert plan.sheet_pixel_count == (
        artifact.sheet_width_pixels * artifact.sheet_height_pixels
    )
    assert plan.sheet_png_byte_count == len(sheet_png)
    assert plan.object_crop_dimensions_pixels == tuple(
        (item.crop_width_pixels, item.crop_height_pixels)
        for item in artifact.objects
    )
    assert plan.within_sheet_pixel_guard is True
    assert (
        plan.maximum_sheet_pixel_count
        == OBJECT_SCENE_ANCHOR_SUPPORT_SHEET_MAX_PIXEL_COUNT
    )

    damaged = deepcopy(plan.to_data())
    damaged["sheet_png_byte_count"] += 1
    _reseal(damaged, "plan_digest")
    with pytest.raises(
        ObjectSceneAnchorSupportSheetError,
        match="geometry or byte identity",
    ):
        ObjectSceneAnchorSupportSheetPlan.from_data(damaged)


def test_sixteen_native_atlas_rows_exceed_transport_without_rendering() -> None:
    _, _, sheet_size = _layout(
        512,
        512,
        ((1, 1, 640, 512),) * 16,
    )

    assert sheet_size == (683, 9020)
    assert object_scene_anchor_grayscale_png_byte_count(*sheet_size) == 6_170_218
    assert object_scene_anchor_grayscale_png_byte_count(*sheet_size) > 4_000_000


def test_native_panel_crop_atlas_and_slot_pixels_are_lossless(
    frozen_sheet,
) -> None:
    panel, inventory, catalog, manifest, artifact, sheet_png = frozen_sheet
    sheet = _png_pixels(sheet_png)
    with Image.open(BytesIO(sheet_png)) as image:
        assert image.size == (
            artifact.sheet_width_pixels,
            artifact.sheet_height_pixels,
        )

    panel_luminance = 255 - _visual._decode_png(panel)
    assert np.array_equal(
        sheet[
            artifact.panel_y_pixels : artifact.panel_y_pixels
            + artifact.panel_height_pixels,
            artifact.panel_x_pixels : artifact.panel_x_pixels
            + artifact.panel_width_pixels,
        ],
        panel_luminance,
    )

    for presentation, entry, decision in zip(
        artifact.objects,
        catalog.entries,
        manifest.object_decisions,
        strict=True,
    ):
        crop_png = render_object_scene_anchor_object_crop(panel, inventory, entry)
        atlas, atlas_png = render_object_scene_anchor_atlas(decision)
        assert atlas_png is not None
        crop = _png_pixels(crop_png)
        atlas_pixels = _png_pixels(atlas_png)
        assert np.array_equal(
            sheet[
                presentation.crop_y_pixels : presentation.crop_y_pixels
                + presentation.crop_height_pixels,
                presentation.crop_x_pixels : presentation.crop_x_pixels
                + presentation.crop_width_pixels,
            ],
            crop,
        )
        assert np.array_equal(
            sheet[
                presentation.atlas_y_pixels : presentation.atlas_y_pixels
                + presentation.atlas_height_pixels,
                presentation.atlas_x_pixels : presentation.atlas_x_pixels
                + presentation.atlas_width_pixels,
            ],
            atlas_pixels,
        )
        for placement, slot in zip(
            presentation.atlas_slots, atlas.slots, strict=True
        ):
            tile = atlas_pixels[
                placement.atlas_row_index * placement.height_pixels : (
                    placement.atlas_row_index + 1
                )
                * placement.height_pixels,
                placement.atlas_column_index * placement.width_pixels : (
                    placement.atlas_column_index + 1
                )
                * placement.width_pixels,
            ]
            assert np.array_equal(
                sheet[
                    placement.sheet_y_pixels : placement.sheet_y_pixels
                    + placement.height_pixels,
                    placement.sheet_x_pixels : placement.sheet_x_pixels
                    + placement.width_pixels,
                ],
                tile,
            )
            assert placement.atlas_slot_digest == slot.slot_digest


def test_receipt_binds_decision_only_component_identities_and_aliases(
    frozen_sheet,
) -> None:
    panel, inventory, catalog, manifest, artifact, _ = frozen_sheet
    data = artifact.to_data()
    keys = tuple(_all_keys(data))

    assert not any("lean" in key.casefold() for key in keys)
    assert not any("entry_digest" in key for key in keys)
    assert not any("salience_artifact" in key for key in keys)
    assert not any("raw_graph" in key for key in keys)
    assert not any("audit_graph" in key for key in keys)
    assert not any("catalog_digest" in key for key in keys)
    assert data["python_is_canonical_authority"] is True
    assert data["object_omission_allowed"] is False
    assert data["component_resampling_allowed"] is False
    assert data["original_panel_png_digest"] == hashlib.sha256(panel).hexdigest()

    for presentation, entry, decision in zip(
        artifact.objects,
        catalog.entries,
        manifest.object_decisions,
        strict=True,
    ):
        crop_png = render_object_scene_anchor_object_crop(panel, inventory, entry)
        atlas, atlas_png = render_object_scene_anchor_atlas(decision)
        assert atlas_png is not None
        assert presentation.decision_manifest_digest == decision.manifest_digest
        assert presentation.crop_png_digest == hashlib.sha256(crop_png).hexdigest()
        assert presentation.atlas_png_digest == hashlib.sha256(atlas_png).hexdigest()
        assert presentation.atlas_artifact_digest == atlas.artifact_digest
        assert presentation.atlas_slot_map_digest == atlas.slot_map_digest

    first_slots = artifact.objects[0].atlas_slots
    assert tuple(
        (item.anchor_kind, item.anchor_id, item.binding_alias)
        for item in first_slots
    ) == (
        ("entity", "entity", "binding_000"),
        ("part", "part-00000000", "binding_000"),
        ("part", "part-00000001", "binding_001"),
        ("frame", "frame-00000000", "binding_000"),
    )


def test_reordering_resealed_omission_and_component_tampering_are_detected(
    frozen_sheet,
) -> None:
    panel, inventory, catalog, manifest, artifact, sheet_png = frozen_sheet

    reordered = deepcopy(artifact.to_data())
    reordered["objects"].reverse()
    _reseal(reordered, "artifact_digest")
    with pytest.raises(ObjectSceneAnchorSupportSheetError, match="objects in order"):
        ObjectSceneAnchorSupportSheet.from_data(reordered)

    omitted = deepcopy(artifact.to_data())
    omitted["proposal_count"] = 1
    omitted["object_ids"] = omitted["object_ids"][:1]
    omitted["objects"] = omitted["objects"][:1]
    _, _, omitted_size = _layout(
        omitted["panel_width_pixels"],
        omitted["panel_height_pixels"],
        tuple(
            (
                item["crop_width_pixels"],
                item["crop_height_pixels"],
                item["atlas_width_pixels"],
                item["atlas_height_pixels"],
            )
            for item in omitted["objects"]
        ),
    )
    omitted["sheet_width_pixels"], omitted["sheet_height_pixels"] = omitted_size
    omitted["sheet_png_byte_count"] = (
        object_scene_anchor_grayscale_png_byte_count(*omitted_size)
    )
    _reseal(omitted, "artifact_digest")
    truncated = ObjectSceneAnchorSupportSheet.from_data(omitted)
    with pytest.raises(ObjectSceneAnchorSupportSheetError, match="exact panel"):
        verify_object_scene_anchor_support_sheet(
            truncated,
            sheet_png,
            panel,
            inventory,
            catalog,
            manifest,
        )

    forged = deepcopy(artifact.to_data())
    forged["objects"][0]["crop_png_digest"] = "0" * 64
    _reseal(forged["objects"][0], "presentation_digest")
    _reseal(forged, "artifact_digest")
    forged_artifact = ObjectSceneAnchorSupportSheet.from_data(forged)
    with pytest.raises(ObjectSceneAnchorSupportSheetError, match="exact panel"):
        verify_object_scene_anchor_support_sheet(
            forged_artifact,
            sheet_png,
            panel,
            inventory,
            catalog,
            manifest,
        )

    changed_sheet = sheet_png[:-1] + bytes((sheet_png[-1] ^ 1,))
    with pytest.raises(ObjectSceneAnchorSupportSheetError, match="exact panel"):
        verify_object_scene_anchor_support_sheet(
            artifact,
            changed_sheet,
            panel,
            inventory,
            catalog,
            manifest,
        )


def test_nonclean_anchor_decision_cannot_create_a_partial_support_sheet() -> None:
    panel = _scene()
    inventory = extract_object_scene_proposal_inventory(panel)
    catalog = extract_object_scene_anchor_catalog(
        panel,
        inventory,
        AnchorSalienceLimits(max_padded_pixels=1),
    )
    manifest = build_object_scene_anchor_panel_decision_manifest(
        catalog, panel, inventory
    )

    assert any(item.salience_state != "clean" for item in manifest.object_decisions)
    with pytest.raises(
        ObjectSceneAnchorSupportSheetError,
        match="clean exhaustive anchor atlas",
    ):
        build_object_scene_anchor_support_sheet(
            panel, inventory, catalog, manifest
        )

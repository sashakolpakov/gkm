from __future__ import annotations

from io import BytesIO

from PIL import Image, ImageDraw
import numpy as np
import pytest

from bongard import prototype_object_hypotheses as hypotheses
from bongard.object_scene_anchor_catalog import extract_object_scene_anchor_catalog
from bongard.object_scene_anchor_crop import (
    ObjectSceneAnchorCropError,
    render_object_scene_anchor_object_crop,
    verify_object_scene_anchor_object_crop,
)
from bongard.object_scene_visual_frontend import (
    extract_object_scene_proposal_inventory,
)


def _scene(*, shift: int = 0) -> bytes:
    image = Image.new("RGB", (72, 48), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((4 + shift, 8, 18 + shift, 25), fill="black")
    draw.line((43, 26, 51, 10, 59, 26), fill="black", width=2)
    output = BytesIO()
    image.save(output, format="PNG", optimize=False)
    return output.getvalue()


def test_full_style_object_crops_are_exact_deterministic_and_native_size() -> None:
    panel = _scene()
    inventory = extract_object_scene_proposal_inventory(panel)
    catalog = extract_object_scene_anchor_catalog(panel, inventory)

    assert catalog.proposal_count == 2
    for entry in catalog.entries:
        first = render_object_scene_anchor_object_crop(
            panel, inventory, entry
        )
        second = render_object_scene_anchor_object_crop(
            panel, inventory, entry
        )
        assert first == second
        assert (
            verify_object_scene_anchor_object_crop(
                first, panel, inventory, entry
            )
            == first
        )
        with Image.open(BytesIO(first)) as image:
            assert image.format == "PNG"
            assert image.mode == "L"
            assert image.size == (
                entry.crop_width_pixels,
                entry.crop_height_pixels,
            )
            luminance = np.ascontiguousarray(
                np.asarray(image, dtype=np.uint8)
            )
        strength = np.ascontiguousarray(255 - luminance, dtype=np.uint8)
        assert (
            hypotheses._crop_pixel_digest(strength)
            == entry.masked_crop_pixel_digest
        )


def test_crop_verifier_rejects_other_object_and_other_panel_pixels() -> None:
    panel = _scene()
    inventory = extract_object_scene_proposal_inventory(panel)
    catalog = extract_object_scene_anchor_catalog(panel, inventory)
    first = render_object_scene_anchor_object_crop(
        panel, inventory, catalog.entries[0]
    )
    second = render_object_scene_anchor_object_crop(
        panel, inventory, catalog.entries[1]
    )

    with pytest.raises(ObjectSceneAnchorCropError, match="exact panel"):
        verify_object_scene_anchor_object_crop(
            second, panel, inventory, catalog.entries[0]
        )
    with pytest.raises(ValueError, match="exact PNG replay"):
        verify_object_scene_anchor_object_crop(
            first, _scene(shift=1), inventory, catalog.entries[0]
        )

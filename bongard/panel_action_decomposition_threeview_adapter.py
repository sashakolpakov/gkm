"""Deterministic raw/crop/edge view for the bounded FIT decomposition ablation.

This is a controlled ablation of the four-quadrant multiview adapter.  It keeps
the raw, square ink crop, and binary inner-boundary quadrants byte-for-byte
under the same preprocessing policy, while replacing the coarse carrier-density
quadrant with exact white.  No density image is generated or exposed.
"""

from __future__ import annotations

from bongard.runtime_source_snapshot import capture_loaded_source, verify_loaded_source


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)

import hashlib
from typing import Any

from PIL import Image, ImageChops, ImageFilter, ImageOps, __version__ as PILLOW_VERSION

from bongard.canonical import canonical_digest
from bongard.panel_action_count_multiview_adapter import (
    BOUNDARY_NEIGHBORHOOD,
    OUTPUT_SIZE,
    _decode_exact_png,
    _ink_mask,
    _png_bytes,
    _square_ink_crop,
    panel_action_count_multiview_adapter_source_digest,
)


PARENT_OUTCOME_DIGEST = (
    "sha256:395c4e3d9c52695fc3f9f5f4c8829f9f270d4d2e60d6ef6c2818f57dcc632488"
)
ALGORITHM_SCHEMA = "gkm.bongard-action-decomposition-threeview-algorithm.v1"
RESULT_SCHEMA = "gkm.bongard-action-decomposition-threeview-result.v1"
ALGORITHM_ID = (
    "bongard.panel-action-decomposition-threeview/raw-crop-edge-blank-python-v1"
)


class ActionDecompositionThreeviewError(ValueError):
    """The source PNG or controlled three-view ablation differs."""


def panel_action_decomposition_threeview_adapter_source_digest() -> str:
    return verify_loaded_source(__name__, expected_source_sha256=_LOADED_SOURCE_SHA256)


def threeview_algorithm_record() -> dict[str, Any]:
    body: dict[str, Any] = {
        "schema": ALGORITHM_SCHEMA,
        "algorithm_id": ALGORITHM_ID,
        "parent_fit_outcome_record_digest": PARENT_OUTCOME_DIGEST,
        "implementation_source_sha256": (
            panel_action_decomposition_threeview_adapter_source_digest()
        ),
        "shared_multiview_adapter_source_sha256": (
            panel_action_count_multiview_adapter_source_digest()
        ),
        "pillow_version": PILLOW_VERSION,
        "shared_preprocessing": {
            "raw": "alpha_composited_on_white_at_original_512_canvas_scale",
            "crop": "same_deterministic_square_ink_crop_as_parent_multiview",
            "edge": "same_binary_inner_boundary_as_parent_multiview",
        },
        "montage": {
            "size": [1024, 1024],
            "quadrants": [
                "top_left_raw",
                "top_right_square_ink_crop",
                "bottom_left_binary_ink_boundary",
                "bottom_right_exact_white_ablation",
            ],
        },
        "coarse_carrier_density_generated": False,
        "coarse_carrier_density_model_visible": False,
        "semantic_action_boundaries_inferred": False,
        "candidate_independent": True,
        "task_concept_independent": True,
        "side_or_role_independent": True,
        "formula_independent": True,
        "python_is_canonical_authority": True,
        "lean_present": False,
        "lean_required": False,
        "lean_removable": True,
        "engineering_only": True,
        "scientific_calibration_supplied": False,
    }
    return {**body, "record_digest": "sha256:" + canonical_digest(body)}


def build_action_decomposition_threeview(
    panel_png: bytes,
) -> tuple[bytes, dict[str, Any]]:
    """Return the fixed three-visible-quadrant montage and provenance record."""

    raw = _decode_exact_png(panel_png)
    source_mask = _ink_mask(raw)
    zoom, bbox, margin, square_bounds = _square_ink_crop(raw, source_mask)
    zoom_mask = _ink_mask(zoom)
    eroded = zoom_mask.filter(ImageFilter.MaxFilter(BOUNDARY_NEIGHBORHOOD))
    boundary = ImageOps.invert(ImageChops.difference(eroded, zoom_mask)).convert("RGB")
    blank = Image.new("RGB", (512, 512), (255, 255, 255))

    montage = Image.new("RGB", OUTPUT_SIZE, (255, 255, 255))
    montage.paste(raw, (0, 0))
    montage.paste(zoom, (512, 0))
    montage.paste(boundary, (0, 512))
    montage.paste(blank, (512, 512))

    raw_png = _png_bytes(raw)
    zoom_png = _png_bytes(zoom)
    boundary_png = _png_bytes(boundary)
    montage_png = _png_bytes(montage)
    algorithm = threeview_algorithm_record()
    body: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "algorithm_record_digest": algorithm["record_digest"],
        "source_png_sha256": hashlib.sha256(panel_png).hexdigest(),
        "source_png_byte_count": len(panel_png),
        "raw_png_sha256": hashlib.sha256(raw_png).hexdigest(),
        "raw_png_byte_count": len(raw_png),
        "ink_bbox": bbox,
        "margin_pixels": margin,
        "square_source_bounds": square_bounds,
        "zoom_png_sha256": hashlib.sha256(zoom_png).hexdigest(),
        "zoom_png_byte_count": len(zoom_png),
        "boundary_png_sha256": hashlib.sha256(boundary_png).hexdigest(),
        "boundary_png_byte_count": len(boundary_png),
        "blank_quadrant_rgb": [255, 255, 255],
        "blank_quadrant_size": [512, 512],
        "montage_png_sha256": hashlib.sha256(montage_png).hexdigest(),
        "montage_png_byte_count": len(montage_png),
        "all_visible_views_derived_only_from_source_png": True,
        "coarse_carrier_density_generated": False,
        "semantic_action_count_inferred": False,
        "candidate_independent": True,
    }
    return montage_png, {**body, "record_digest": "sha256:" + canonical_digest(body)}


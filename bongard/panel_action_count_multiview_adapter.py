"""Deterministic, target-independent multiview panels for action counting.

The adapter turns one exact 512 by 512 Bongard PNG into one 1024 by 1024
model image with four fixed quadrants:

* the unmodified, alpha-composited source;
* a square crop around all visible ink;
* a one-pixel binary ink-boundary view of that crop; and
* a coarse carrier-density view that expands and softly connects nearby ink.

The latter two views are intentionally observations, not semantic labels.  In
particular, they do not decide where a generator action begins or ends.  Every
byte-level source/derived relationship is returned in a content-addressed
record so a phase runner can bind it before any labels are opened.
"""

from __future__ import annotations

from bongard.runtime_source_snapshot import capture_loaded_source, verify_loaded_source


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)

from io import BytesIO
import hashlib
from typing import Any

from PIL import Image, ImageChops, ImageFilter, ImageOps, __version__ as PILLOW_VERSION

from bongard.canonical import canonical_digest


MULTIVIEW_ALGORITHM_SCHEMA = "gkm.bongard-action-count-multiview-algorithm.v1"
MULTIVIEW_RESULT_SCHEMA = "gkm.bongard-action-count-multiview-result.v1"
MULTIVIEW_ALGORITHM_ID = (
    "bongard.panel-action-count-multiview/raw-crop-edge-carrier-python-v1"
)
INPUT_SIZE = (512, 512)
OUTPUT_SIZE = (1024, 1024)
INK_THRESHOLD = 245
MIN_MARGIN_PIXELS = 8
MARGIN_DIVISOR = 8
BOUNDARY_NEIGHBORHOOD = 3
CARRIER_DILATION_NEIGHBORHOOD = 9
CARRIER_BLUR_RADIUS = 5
PNG_COMPRESS_LEVEL = 9


class ActionCountMultiviewError(ValueError):
    """The source PNG or deterministic transform is invalid."""


def panel_action_count_multiview_adapter_source_digest() -> str:
    """Return the authenticated bytes of this loaded adapter."""

    return verify_loaded_source(__name__, expected_source_sha256=_LOADED_SOURCE_SHA256)


def multiview_algorithm_record() -> dict[str, Any]:
    """Return the closed algorithm record bound by every derived view."""

    body: dict[str, Any] = {
        "schema": MULTIVIEW_ALGORITHM_SCHEMA,
        "algorithm_id": MULTIVIEW_ALGORITHM_ID,
        "implementation_source_sha256": (
            panel_action_count_multiview_adapter_source_digest()
        ),
        "pillow_version": PILLOW_VERSION,
        "accepted_input": {"format": "PNG", "mode": "any", "size": [512, 512]},
        "alpha_composite_background_rgb": [255, 255, 255],
        "ink_mask": "minimum_rgb_channel_less_than_245",
        "crop": {
            "bounding_box": "left_top_inclusive_right_bottom_exclusive",
            "margin_pixels": "max(8,ceil(max(width,height)/8))",
            "square_centering": (
                "floor_divide_leftover_equally_extra_pixel_right_or_bottom"
            ),
            "out_of_source_padding_rgb": [255, 255, 255],
            "resize": [512, 512],
            "resampler": "Pillow.Resampling.LANCZOS",
        },
        "boundary": {
            "input": "thresholded_zoomed_ink_mask",
            "operation": "foreground_inner_boundary_via_3x3_max_filter",
            "foreground_rgb": [0, 0, 0],
            "background_rgb": [255, 255, 255],
        },
        "carrier_density": {
            "input": "thresholded_zoomed_ink_mask",
            "operation": "9x9_dark_min_filter_then_gaussian_blur_radius_5",
            "semantic_action_boundaries_inferred": False,
            "purpose": "make_nearby_decorated_ink_alignment_visible",
        },
        "montage": {
            "size": [1024, 1024],
            "quadrants": [
                "top_left_raw",
                "top_right_square_ink_crop",
                "bottom_left_binary_ink_boundary",
                "bottom_right_coarse_carrier_density",
            ],
            "separator": "none",
        },
        "output": {
            "mode": "RGB",
            "format": "PNG",
            "compress_level": PNG_COMPRESS_LEVEL,
            "optimize": False,
            "metadata": "none",
        },
        "candidate_independent": True,
        "task_concept_independent": True,
        "side_or_role_independent": True,
        "formula_independent": True,
        "semantic_action_count_inferred": False,
        "python_is_canonical_authority": True,
        "lean_present": False,
        "lean_required": False,
        "lean_removable": True,
        "engineering_only": True,
        "scientific_calibration_supplied": False,
    }
    return {**body, "record_digest": "sha256:" + canonical_digest(body)}


def _png_bytes(image: Image.Image) -> bytes:
    output = BytesIO()
    image.save(
        output,
        format="PNG",
        compress_level=PNG_COMPRESS_LEVEL,
        optimize=False,
    )
    return output.getvalue()


def _decode_exact_png(panel_png: bytes) -> Image.Image:
    if type(panel_png) is not bytes or not panel_png.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ActionCountMultiviewError("source is not exact PNG bytes")
    try:
        with Image.open(BytesIO(panel_png)) as decoded:
            if decoded.format != "PNG" or decoded.size != INPUT_SIZE:
                raise ActionCountMultiviewError(
                    "source must be an exact 512 by 512 PNG"
                )
            decoded.load()
            rgba = decoded.convert("RGBA")
    except ActionCountMultiviewError:
        raise
    except Exception as exc:
        raise ActionCountMultiviewError("source PNG cannot be decoded") from exc
    background = Image.new("RGBA", INPUT_SIZE, (255, 255, 255, 255))
    background.alpha_composite(rgba)
    return background.convert("RGB")


def _ink_mask(rgb: Image.Image) -> Image.Image:
    mask = Image.new("L", rgb.size, 255)
    mask.putdata(
        [
            0 if min(pixel) < INK_THRESHOLD else 255
            for pixel in tuple(rgb.get_flattened_data())
        ]
    )
    return mask


def _square_ink_crop(rgb: Image.Image, mask: Image.Image) -> tuple[Image.Image, list[int], int, list[int]]:
    # getbbox treats nonzero pixels as foreground, whereas this module stores
    # ink as zero so it can directly display the mask as black-on-white.
    bbox = ImageOps.invert(mask).getbbox()
    if bbox is None:
        raise ActionCountMultiviewError("source contains no visible ink")
    width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]
    margin = max(MIN_MARGIN_PIXELS, (max(width, height) + MARGIN_DIVISOR - 1) // MARGIN_DIVISOR)
    square_side = max(width, height) + 2 * margin
    square_left = (bbox[0] + bbox[2] - square_side) // 2
    square_top = (bbox[1] + bbox[3] - square_side) // 2
    square_right = square_left + square_side
    square_bottom = square_top + square_side
    source_left, source_top = max(0, square_left), max(0, square_top)
    source_right, source_bottom = min(512, square_right), min(512, square_bottom)
    canvas = Image.new("RGB", (square_side, square_side), (255, 255, 255))
    canvas.paste(
        rgb.crop((source_left, source_top, source_right, source_bottom)),
        (source_left - square_left, source_top - square_top),
    )
    zoom = canvas.resize(INPUT_SIZE, Image.Resampling.LANCZOS)
    return zoom, list(bbox), margin, [square_left, square_top, square_right, square_bottom]


def build_action_count_multiview(panel_png: bytes) -> tuple[bytes, dict[str, Any]]:
    """Build and content-address one deterministic four-quadrant model view."""

    raw = _decode_exact_png(panel_png)
    source_mask = _ink_mask(raw)
    zoom, bbox, margin, square_bounds = _square_ink_crop(raw, source_mask)
    zoom_mask = _ink_mask(zoom)

    # For black foreground, MaxFilter erodes the foreground.  The positive
    # difference therefore selects the one-pixel foreground inner boundary;
    # invert it to return the common black-on-white convention.
    eroded = zoom_mask.filter(ImageFilter.MaxFilter(BOUNDARY_NEIGHBORHOOD))
    boundary = ImageOps.invert(ImageChops.difference(eroded, zoom_mask)).convert("RGB")

    # Expand black ink before low-pass smoothing.  Marker chains and zigzags
    # then expose their coarse alignment while all original evidence remains
    # available in the raw and zoom quadrants.
    carrier = zoom_mask.filter(
        ImageFilter.MinFilter(CARRIER_DILATION_NEIGHBORHOOD)
    ).filter(ImageFilter.GaussianBlur(CARRIER_BLUR_RADIUS)).convert("RGB")

    montage = Image.new("RGB", OUTPUT_SIZE, (255, 255, 255))
    montage.paste(raw, (0, 0))
    montage.paste(zoom, (512, 0))
    montage.paste(boundary, (0, 512))
    montage.paste(carrier, (512, 512))

    raw_rendered_png = _png_bytes(raw)
    zoom_png = _png_bytes(zoom)
    boundary_png = _png_bytes(boundary)
    carrier_png = _png_bytes(carrier)
    montage_png = _png_bytes(montage)
    algorithm = multiview_algorithm_record()
    body: dict[str, Any] = {
        "schema": MULTIVIEW_RESULT_SCHEMA,
        "algorithm_record_digest": algorithm["record_digest"],
        "source_png_sha256": hashlib.sha256(panel_png).hexdigest(),
        "source_png_byte_count": len(panel_png),
        "alpha_composited_raw_png_sha256": hashlib.sha256(raw_rendered_png).hexdigest(),
        "alpha_composited_raw_png_byte_count": len(raw_rendered_png),
        "ink_bbox": bbox,
        "margin_pixels": margin,
        "square_source_bounds": square_bounds,
        "zoom_png_sha256": hashlib.sha256(zoom_png).hexdigest(),
        "zoom_png_byte_count": len(zoom_png),
        "boundary_png_sha256": hashlib.sha256(boundary_png).hexdigest(),
        "boundary_png_byte_count": len(boundary_png),
        "carrier_density_png_sha256": hashlib.sha256(carrier_png).hexdigest(),
        "carrier_density_png_byte_count": len(carrier_png),
        "montage_png_sha256": hashlib.sha256(montage_png).hexdigest(),
        "montage_png_byte_count": len(montage_png),
        "all_views_derived_only_from_source_png": True,
        "semantic_action_count_inferred": False,
        "candidate_independent": True,
    }
    return montage_png, {**body, "record_digest": "sha256:" + canonical_digest(body)}

"""Synthetic-only four-scale pooled shape control.

This module preserves one historical 112-float image representation solely as
a neutral control for process-issued synthetic panels. It contains no corpus,
labels, estimator, fitting, prediction, command-line, or execution-authority
surface. The default extractor requires the original bounded renderer; a
separate callback seam permits another synthetic-only renderer to authenticate
bytes without exposing its labels, targets, carriers, or programs.
"""

from __future__ import annotations

from bongard.runtime_source_snapshot import capture_loaded_source, verify_loaded_source


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)

from io import BytesIO
import hashlib
import importlib.metadata
import math
import platform
import sys
from typing import Final

import numpy as np
from PIL import Image
from scipy import ndimage

from bongard import panel_action_count_synthetic_identifiability as synthetic
from bongard import runtime_source_snapshot


IMAGE_SIZE: Final = 64
MAX_PNG_BYTES: Final = 16 * 1024 * 1024
MAX_PNG_DIMENSION: Final = 2048
MAX_PNG_PIXELS: Final = 4_194_304

SCALE_SPECS: Final = (
    ("raw_threshold_10_of_255", 0.0, 10.0 / 255.0, False),
    ("raw_threshold_then_3x3_closing_once", 0.0, 10.0 / 255.0, True),
    ("gaussian_sigma_1p5_threshold_0p08", 1.5, 0.08, False),
    ("gaussian_sigma_3p0_threshold_0p035", 3.0, 0.035, False),
)

PER_SCALE_FEATURE_NAMES: Final = (
    "foreground_area_fraction",
    "boundary_area_fraction",
    "component_count_div_32",
    "hole_count_div_32",
    "largest_component_fraction",
    "component_area_sd_div_foreground_area",
    "bbox_height_fraction",
    "bbox_width_fraction",
    "bbox_aspect_ratio",
    "centroid_y_fraction",
    "centroid_x_fraction",
    "second_moment_anisotropy",
    "skeleton_area_fraction",
    "endpoint_cluster_count_div_32",
    "branch_cluster_count_div_32",
    "isolated_skeleton_pixel_count_div_32",
    "eight_neighbor_raster_cycle_rank_div_32",
    "skeleton_edge_count_fraction",
    "edge_orientation_horizontal_fraction",
    "edge_orientation_diagonal_up_fraction",
    "edge_orientation_vertical_fraction",
    "edge_orientation_diagonal_down_fraction",
    "degree_two_turn_cos_le_neg_0p9_fraction",
    "degree_two_turn_cos_neg_0p9_to_neg_0p25_fraction",
    "degree_two_turn_cos_neg_0p25_to_0p25_fraction",
    "degree_two_turn_cos_gt_0p25_fraction",
    "mean_skeleton_half_width_div_8",
    "max_skeleton_half_width_div_8",
)
FEATURE_NAMES: Final = tuple(
    f"{scale_name}:{feature_name}"
    for scale_name, _sigma, _threshold, _closing in SCALE_SPECS
    for feature_name in PER_SCALE_FEATURE_NAMES
)

_DIRECTIONS: Final = (
    (-1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
)


class SyntheticPooledControlError(ValueError):
    """The synthetic provenance boundary or fixed representation differs."""


def source_sha256() -> str:
    """Verify and return the import-time source address."""

    return verify_loaded_source(__name__, expected_source_sha256=_LOADED_SOURCE_SHA256)


def _authority_preflight() -> None:
    source_sha256()
    if (
        type(SCALE_SPECS) is not tuple
        or len(SCALE_SPECS) != 4
        or type(PER_SCALE_FEATURE_NAMES) is not tuple
        or len(PER_SCALE_FEATURE_NAMES) != 28
        or len(set(PER_SCALE_FEATURE_NAMES)) != 28
        or type(FEATURE_NAMES) is not tuple
        or len(FEATURE_NAMES) != 112
        or len(set(FEATURE_NAMES)) != 112
    ):
        raise SyntheticPooledControlError("feature vocabulary differs")
    if any(type(name) is not str or not name for name in FEATURE_NAMES):
        raise SyntheticPooledControlError("feature vocabulary types differ")


def _preprocess_png_bytes(raw: bytes) -> np.ndarray:
    """Tight-crop an issued PNG to the fixed 64x64 uint8 ink plane."""

    if type(raw) is not bytes or not 0 < len(raw) <= MAX_PNG_BYTES:
        raise SyntheticPooledControlError("PNG byte count is outside the fixed cap")
    try:
        with Image.open(BytesIO(raw)) as image:
            if image.format != "PNG" or getattr(image, "n_frames", 1) != 1:
                raise SyntheticPooledControlError("input must be one PNG frame")
            if (
                type(image.width) is not int
                or type(image.height) is not int
                or image.width <= 0
                or image.height <= 0
                or image.width > MAX_PNG_DIMENSION
                or image.height > MAX_PNG_DIMENSION
                or image.width * image.height > MAX_PNG_PIXELS
            ):
                raise SyntheticPooledControlError("PNG dimensions exceed the fixed cap")
            image.load()
            gray = np.asarray(image.convert("L"), dtype=np.uint8)
    except SyntheticPooledControlError:
        raise
    except Exception as exc:  # pragma: no cover - decoder/environment failure
        raise SyntheticPooledControlError(f"cannot decode PNG: {exc}") from exc
    if gray.ndim != 2 or gray.size <= 0:
        raise SyntheticPooledControlError("decoded PNG plane differs")
    ys, xs = np.nonzero(gray < 250)
    if len(xs) == 0:
        raise SyntheticPooledControlError("PNG has no ink")
    crop = gray[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    height, width = crop.shape
    margin = math.ceil(0.08 * max(height, width))
    side = max(height, width) + 2 * margin
    canvas = np.full((side, side), 255, dtype=np.uint8)
    top, left = (side - height) // 2, (side - width) // 2
    canvas[top : top + height, left : left + width] = crop
    resized = Image.fromarray(canvas, mode="L").resize(
        (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR
    )
    ink = np.ascontiguousarray(255 - np.asarray(resized, dtype=np.uint8))
    if ink.shape != (IMAGE_SIZE, IMAGE_SIZE) or ink.dtype != np.uint8:
        raise SyntheticPooledControlError("preprocessed ink plane differs")
    return ink


def _zhang_suen(mask: np.ndarray) -> np.ndarray:
    """Deterministic topology-preserving thinning until a fixed point."""

    if type(mask) is not np.ndarray or mask.shape != (IMAGE_SIZE, IMAGE_SIZE):
        raise SyntheticPooledControlError("thinning mask shape differs")
    current = np.ascontiguousarray(mask, dtype=bool)
    if not current.any():
        return current
    ys, xs = np.nonzero(current)
    cropped = current[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    work = np.pad(cropped, 1, constant_values=False)
    for _ in range(max(work.shape)):
        changed = False
        for phase in (0, 1):
            padded = np.pad(work, 1, constant_values=False)
            neighbours = (
                padded[:-2, 1:-1],
                padded[:-2, 2:],
                padded[1:-1, 2:],
                padded[2:, 2:],
                padded[2:, 1:-1],
                padded[2:, :-2],
                padded[1:-1, :-2],
                padded[:-2, :-2],
            )
            count = sum(item.astype(np.uint8) for item in neighbours)
            transitions = sum(
                ((~neighbours[index]) & neighbours[(index + 1) % 8]).astype(np.uint8)
                for index in range(8)
            )
            p2, _p3, p4, _p5, p6, _p7, p8, _p9 = neighbours
            if phase == 0:
                gate_a, gate_b = ~(p2 & p4 & p6), ~(p4 & p6 & p8)
            else:
                gate_a, gate_b = ~(p2 & p4 & p8), ~(p2 & p6 & p8)
            delete = (
                work
                & (count >= 2)
                & (count <= 6)
                & (transitions == 1)
                & gate_a
                & gate_b
            )
            if delete.any():
                work &= ~delete
                changed = True
        if not changed:
            break
    else:  # pragma: no cover - convergence guard
        raise SyntheticPooledControlError("thinning exceeded its dimension bound")
    result = np.zeros_like(current)
    result[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1] = work[1:-1, 1:-1]
    return np.ascontiguousarray(result)


def _one_scale_features(mask: np.ndarray) -> np.ndarray:
    if type(mask) is not np.ndarray or mask.shape != (IMAGE_SIZE, IMAGE_SIZE):
        raise SyntheticPooledControlError("feature mask shape differs")
    s8 = np.ones((3, 3), dtype=np.uint8)
    s4 = np.asarray(((0, 1, 0), (1, 1, 1), (0, 1, 0)), dtype=np.uint8)
    unit_directions = np.asarray(_DIRECTIONS, dtype=np.float64)
    unit_directions /= np.linalg.norm(unit_directions, axis=1)[:, None]
    mask = np.ascontiguousarray(mask, dtype=bool)
    area = int(mask.sum())
    labels, component_count = ndimage.label(mask, structure=s8)
    sizes = np.bincount(labels.ravel())[1:]
    holes = ndimage.binary_fill_holes(mask) & ~mask
    _hole_labels, hole_count = ndimage.label(holes, structure=s4)
    boundary = mask & ~ndimage.binary_erosion(mask, structure=s8, border_value=0)
    ys, xs = np.nonzero(mask)
    if area:
        height, width = int(ys.max() - ys.min() + 1), int(xs.max() - xs.min() + 1)
        centroid_y, centroid_x = float(ys.mean() / 63), float(xs.mean() / 63)
        centered = np.stack(((ys - ys.mean()) / 64, (xs - xs.mean()) / 64))
        covariance = np.cov(centered, bias=True) if area > 1 else np.zeros((2, 2))
        eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0)
        anisotropy = float(eigenvalues[-1] / (eigenvalues.sum() + 1e-9))
    else:  # Valid issued panels remain nonempty at every fixed scale.
        height = width = 0
        centroid_y = centroid_x = anisotropy = 0.0
    skeleton = _zhang_suen(mask)
    vertex_count = int(skeleton.sum())
    _component_labels, skeleton_components = ndimage.label(skeleton, structure=s8)
    padded = np.pad(skeleton, 1)
    neighbours = [
        padded[1 + dy : 65 + dy, 1 + dx : 65 + dx]
        for dy, dx in _DIRECTIONS
    ]
    degree = sum(item.astype(np.uint8) for item in neighbours)
    endpoints = skeleton & (degree == 1)
    branches = skeleton & (degree >= 3)
    isolated = skeleton & (degree == 0)
    _labels, endpoint_clusters = ndimage.label(endpoints, structure=s8)
    _labels, branch_clusters = ndimage.label(branches, structure=s8)
    edges = np.asarray(
        (
            np.count_nonzero(skeleton[:, :-1] & skeleton[:, 1:]),
            np.count_nonzero(skeleton[:-1, 1:] & skeleton[1:, :-1]),
            np.count_nonzero(skeleton[:-1, :] & skeleton[1:, :]),
            np.count_nonzero(skeleton[:-1, :-1] & skeleton[1:, 1:]),
        ),
        dtype=np.float64,
    )
    edge_count = float(edges.sum())
    orientations = edges / (edge_count + 1e-9)
    turns = np.zeros(4, dtype=np.float64)
    degree_two = skeleton & (degree == 2)
    for first in range(8):
        for second in range(first + 1, 8):
            count = np.count_nonzero(
                degree_two & neighbours[first] & neighbours[second]
            )
            cosine = float(np.dot(unit_directions[first], unit_directions[second]))
            bucket = (
                0
                if cosine <= -0.9
                else 1
                if cosine <= -0.25
                else 2
                if cosine <= 0.25
                else 3
            )
            turns[bucket] += count
    turns /= float(degree_two.sum()) + 1e-9
    widths = ndimage.distance_transform_edt(mask)[skeleton]
    cycle_rank = max(0.0, edge_count - vertex_count + float(skeleton_components))
    values = (
        area / 4096,
        float(boundary.sum()) / 4096,
        component_count / 32,
        hole_count / 32,
        float(sizes.max()) / area if len(sizes) and area else 0,
        float(np.std(sizes)) / area if len(sizes) and area else 0,
        height / 64,
        width / 64,
        height / (width + 1e-9),
        centroid_y,
        centroid_x,
        anisotropy,
        vertex_count / 4096,
        endpoint_clusters / 32,
        branch_clusters / 32,
        float(isolated.sum()) / 32,
        cycle_rank / 32,
        edge_count / 4096,
        *orientations.tolist(),
        *turns.tolist(),
        (float(widths.mean()) if len(widths) else 0) / 8,
        (float(widths.max()) if len(widths) else 0) / 8,
    )
    result = np.asarray(values, dtype=np.float32)
    if result.shape != (28,) or result.dtype != np.float32 or not np.isfinite(result).all():
        raise SyntheticPooledControlError("feature vector is nonfinite or malformed")
    return result


def extract_feature_vector(png_bytes: bytes) -> np.ndarray:
    """Return 112 pooled floats for one process-issued synthetic PNG."""

    _authority_preflight()
    if type(png_bytes) is not bytes:
        raise SyntheticPooledControlError("synthetic PNG payload must be exact bytes")
    try:
        synthetic.require_issued_synthetic_png(png_bytes)
    except (TypeError, ValueError) as exc:
        raise SyntheticPooledControlError(str(exc)) from exc
    ink = _preprocess_png_bytes(png_bytes)
    raw = ink >= 10
    if not raw.any():
        raise SyntheticPooledControlError("PNG has no ink at the fixed threshold")
    strength = ink.astype(np.float32) / 255.0
    s8 = np.ones((3, 3), dtype=np.uint8)
    masks = (
        raw,
        ndimage.binary_closing(raw, structure=s8, iterations=1),
        ndimage.gaussian_filter(strength, 1.5, mode="constant") >= 0.08,
        ndimage.gaussian_filter(strength, 3.0, mode="constant") >= 0.035,
    )
    result = np.concatenate(tuple(_one_scale_features(mask) for mask in masks))
    if (
        result.shape != (len(FEATURE_NAMES),)
        or result.dtype != np.float32
        or not result.flags.c_contiguous
        or not np.isfinite(result).all()
    ):
        raise SyntheticPooledControlError("multiscale feature vector differs")
    return np.ascontiguousarray(result, dtype=np.float32)


def extract_issued_feature_vector(
    png_bytes: bytes, *, require_issued: object
) -> np.ndarray:
    """Return the same vector after an explicit synthetic issuer validates it.

    This narrow injection seam lets later, separately scoped synthetic
    grammars reuse the neutral representation without teaching this module
    about their record types.  The callback may authenticate bytes only; it
    receives no label, target, carrier, split, or generator program.
    """

    _authority_preflight()
    if type(png_bytes) is not bytes or not callable(require_issued):
        raise SyntheticPooledControlError(
            "issued synthetic feature input differs"
        )
    try:
        digest = require_issued(png_bytes)
    except (TypeError, ValueError) as exc:
        raise SyntheticPooledControlError(str(exc)) from exc
    expected = "sha256:" + hashlib.sha256(png_bytes).hexdigest()
    if type(digest) is not str or digest != expected:
        raise SyntheticPooledControlError(
            "synthetic issuer returned the wrong PNG address"
        )
    ink = _preprocess_png_bytes(png_bytes)
    raw = ink >= 10
    if not raw.any():
        raise SyntheticPooledControlError("PNG has no ink at the fixed threshold")
    strength = ink.astype(np.float32) / 255.0
    s8 = np.ones((3, 3), dtype=np.uint8)
    masks = (
        raw,
        ndimage.binary_closing(raw, structure=s8, iterations=1),
        ndimage.gaussian_filter(strength, 1.5, mode="constant") >= 0.08,
        ndimage.gaussian_filter(strength, 3.0, mode="constant") >= 0.035,
    )
    result = np.concatenate(tuple(_one_scale_features(mask) for mask in masks))
    if (
        result.shape != (len(FEATURE_NAMES),)
        or result.dtype != np.float32
        or not result.flags.c_contiguous
        or not np.isfinite(result).all()
    ):
        raise SyntheticPooledControlError("multiscale feature vector differs")
    return np.ascontiguousarray(result, dtype=np.float32)


def runtime_fingerprint() -> dict[str, str]:
    """Return only runtime versions capable of changing the representation."""

    _authority_preflight()
    value = {
        "byteorder": sys.byteorder,
        "machine": platform.machine(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "pillow": importlib.metadata.version("Pillow"),
        "numpy": importlib.metadata.version("numpy"),
        "scipy": importlib.metadata.version("scipy"),
        "scikit_learn": importlib.metadata.version("scikit-learn"),
    }
    if set(value) != {
        "byteorder", "machine", "python", "platform", "pillow", "numpy",
        "scipy", "scikit_learn",
    } or any(
        type(item) is not str or not item for item in value.values()
    ):
        raise SyntheticPooledControlError("runtime fingerprint differs")
    return value


def dependency_source_addresses() -> dict[str, str]:
    """Bind the only project sources in this synthetic control boundary."""

    _authority_preflight()
    value = {
        "bongard.panel_action_count_synthetic_pooled_control":
            "sha256:" + source_sha256(),
        "bongard.panel_action_count_synthetic_identifiability":
            "sha256:" + synthetic.source_sha256(),
        "bongard.runtime_source_snapshot": "sha256:"
        + runtime_source_snapshot.verify_loaded_source(
            runtime_source_snapshot.__name__,
            expected_source_sha256=(
                runtime_source_snapshot.RUNTIME_SOURCE_SNAPSHOT_SHA256
            ),
        ),
    }
    if any(
        type(address) is not str
        or len(address) != 71
        or not address.startswith("sha256:")
        for address in value.values()
    ):
        raise SyntheticPooledControlError("dependency source address differs")
    return value


__all__ = (
    "FEATURE_NAMES",
    "IMAGE_SIZE",
    "PER_SCALE_FEATURE_NAMES",
    "SCALE_SPECS",
    "SyntheticPooledControlError",
    "dependency_source_addresses",
    "extract_feature_vector",
    "extract_issued_feature_vector",
    "runtime_fingerprint",
    "source_sha256",
)

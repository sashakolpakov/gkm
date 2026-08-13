"""Target-independent analysis-by-synthesis for connected synthetic panels.

The fitter in this module sees only an *issued* PNG and the fixed primitive
catalog published by :mod:`panel_action_count_connected_synthetic`.  It never
consults a declared program, provenance, count label, or canonical/exact-cover
target.  Candidate primitive masks are first required to be subsets of the
observed foreground (which proves that they explain no background pixel), and
then a bounded global set-cover search requires their union to equal every
foreground pixel.  All minimum-cardinality covers through nine primitives are
retained after collapsing only same-kind, byte-identical catalog masks.

This is deliberately a raw synthetic observer, not a semantic claim about
official Bongard data.  Its hypotheses identify visible catalog geometries;
they do not recover otherwise invisible generator history.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from io import BytesIO
import math
from functools import lru_cache
from typing import Final, Literal

import numpy as np
from PIL import Image

from bongard.runtime_source_snapshot import capture_loaded_source, verify_loaded_source
from bongard.canonical import canonical_digest
from bongard import panel_action_count_connected_synthetic as connected
from bongard import panel_action_count_ordered_path_inversion as ordered


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)

IMAGE_SIZE: Final = 64
PIXEL_COUNT: Final = IMAGE_SIZE * IMAGE_SIZE
MAX_VISIBLE_PRIMITIVES: Final = 9
_Kind = Literal["line", "arc"]


class ConnectedSynthesisError(ValueError):
    """An issued transport, catalog, or exact-cover invariant differs."""


def source_sha256() -> str:
    """Verify and return this module's import-time source address."""

    return verify_loaded_source(__name__, expected_source_sha256=_LOADED_SOURCE_SHA256)


def _exact_pixel_tuple(value: object, name: str) -> tuple[int, ...]:
    if (
        type(value) is not tuple
        or any(type(pixel) is not int for pixel in value)
        or value != tuple(sorted(set(value)))
        or (value and (value[0] < 0 or value[-1] >= PIXEL_COUNT))
    ):
        raise ConnectedSynthesisError(f"{name} differs")
    return value


def _exact_yx_tuple(value: object, name: str) -> tuple[tuple[int, int], ...]:
    if (
        type(value) is not tuple
        or any(
            type(point) is not tuple
            or len(point) != 2
            or any(type(coordinate) is not int for coordinate in point)
            or any(not 0 <= coordinate < IMAGE_SIZE for coordinate in point)
            for point in value
        )
        or value != tuple(sorted(set(value)))
    ):
        raise ConnectedSynthesisError(f"{name} differs")
    return value


@dataclass(frozen=True, slots=True)
class ChosenCatalogPrimitive:
    """One materially distinct catalog mask in an exact reconstruction."""

    primitive_id: str
    equivalent_primitive_ids: tuple[str, ...]
    kind: _Kind
    ink_pixels: tuple[int, ...]
    boundary_pixels_yx: tuple[tuple[int, int], ...]
    endpoints_yx: tuple[tuple[int, int], ...]
    path_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if type(self.primitive_id) is not str or not self.primitive_id:
            raise ConnectedSynthesisError("chosen primitive id differs")
        if (
            type(self.equivalent_primitive_ids) is not tuple
            or not self.equivalent_primitive_ids
            or any(type(item) is not str or not item for item in self.equivalent_primitive_ids)
            or self.equivalent_primitive_ids != tuple(sorted(set(self.equivalent_primitive_ids)))
            or self.primitive_id != self.equivalent_primitive_ids[0]
        ):
            raise ConnectedSynthesisError("equivalent primitive ids differ")
        if type(self.kind) is not str or self.kind not in ("line", "arc"):
            raise ConnectedSynthesisError("chosen primitive kind differs")
        if not _exact_pixel_tuple(self.ink_pixels, "chosen primitive ink"):
            raise ConnectedSynthesisError("chosen primitive has no ink")
        _exact_yx_tuple(self.boundary_pixels_yx, "chosen primitive boundary")
        _exact_yx_tuple(self.endpoints_yx, "chosen primitive endpoints")
        if (
            type(self.path_ids) is not tuple
            or any(type(item) is not int or item < 0 for item in self.path_ids)
            or self.path_ids != tuple(sorted(set(self.path_ids)))
        ):
            raise ConnectedSynthesisError("chosen primitive path ids differ")

    @property
    def mask_sha256(self) -> str:
        payload = b"".join(pixel.to_bytes(2, "big") for pixel in self.ink_pixels)
        return "sha256:" + hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class ConnectedProgramHypothesis:
    """One minimum-cardinality, all-pixel catalog explanation."""

    straight_count: int
    arc_count: int
    primitives: tuple[ChosenCatalogPrimitive, ...]
    reconstructed_ink_pixels: tuple[int, ...]
    xor_pixel_count: int
    intersection_over_union: float

    def __post_init__(self) -> None:
        if (
            type(self.straight_count) is not int
            or type(self.arc_count) is not int
            or not 1 <= self.straight_count + self.arc_count <= MAX_VISIBLE_PRIMITIVES
        ):
            raise ConnectedSynthesisError("hypothesis count pair differs")
        if (
            type(self.primitives) is not tuple
            or len(self.primitives) != self.straight_count + self.arc_count
            or any(type(item) is not ChosenCatalogPrimitive for item in self.primitives)
        ):
            raise ConnectedSynthesisError("hypothesis primitive inventory differs")
        if tuple(item.primitive_id for item in self.primitives) != tuple(
            sorted(item.primitive_id for item in self.primitives)
        ):
            raise ConnectedSynthesisError("hypothesis primitive order differs")
        if (
            sum(item.kind == "line" for item in self.primitives) != self.straight_count
            or sum(item.kind == "arc" for item in self.primitives) != self.arc_count
        ):
            raise ConnectedSynthesisError("hypothesis primitive kinds differ")
        _exact_pixel_tuple(self.reconstructed_ink_pixels, "reconstructed ink")
        expected = tuple(
            sorted({pixel for item in self.primitives for pixel in item.ink_pixels})
        )
        if self.reconstructed_ink_pixels != expected:
            raise ConnectedSynthesisError("hypothesis union differs")
        if type(self.xor_pixel_count) is not int or self.xor_pixel_count != 0:
            raise ConnectedSynthesisError("hypothesis is not an exact XOR reconstruction")
        if (
            type(self.intersection_over_union) is not float
            or not math.isfinite(self.intersection_over_union)
            or self.intersection_over_union != 1.0
        ):
            raise ConnectedSynthesisError("hypothesis is not an exact IoU reconstruction")

    @property
    def pair(self) -> tuple[int, int]:
        return self.straight_count, self.arc_count

    @property
    def primitive_ids(self) -> tuple[str, ...]:
        return tuple(item.primitive_id for item in self.primitives)

    @property
    def primitive_kinds(self) -> tuple[_Kind, ...]:
        return tuple(item.kind for item in self.primitives)

    @property
    def geometry_key(self) -> tuple[tuple[str, tuple[int, ...]], ...]:
        return tuple(sorted((item.kind, item.ink_pixels) for item in self.primitives))


@dataclass(frozen=True, slots=True)
class ConnectedFitOutcome:
    """Complete raw result, including ordered graph and exact-cover evidence."""

    disposition: Literal["IDENTIFIED", "AMBIGUOUS", "GAP"]
    reason: str | None
    paths: tuple[ordered.OrderedGraphPath, ...]
    boundary_pixels_yx: tuple[tuple[int, int], ...]
    endpoints_yx: tuple[tuple[int, int], ...]
    foreground_pixel_count: int
    skeleton_pixel_count: int
    minimum_primitive_count: int | None
    hypotheses: tuple[ConnectedProgramHypothesis, ...]

    def __post_init__(self) -> None:
        if type(self.disposition) is not str or self.disposition not in {
            "IDENTIFIED", "AMBIGUOUS", "GAP"
        }:
            raise ConnectedSynthesisError("outcome disposition differs")
        if (
            type(self.paths) is not tuple
            or any(type(item) is not ordered.OrderedGraphPath for item in self.paths)
        ):
            raise ConnectedSynthesisError("outcome graph paths differ")
        _exact_yx_tuple(self.boundary_pixels_yx, "outcome boundary")
        _exact_yx_tuple(self.endpoints_yx, "outcome endpoints")
        if (
            type(self.foreground_pixel_count) is not int
            or type(self.skeleton_pixel_count) is not int
            or self.foreground_pixel_count <= 0
            or self.skeleton_pixel_count <= 0
        ):
            raise ConnectedSynthesisError("outcome pixel counts differ")
        if (
            type(self.hypotheses) is not tuple
            or any(type(item) is not ConnectedProgramHypothesis for item in self.hypotheses)
        ):
            raise ConnectedSynthesisError("outcome hypotheses differ")
        for item in self.hypotheses:
            ConnectedProgramHypothesis.__post_init__(item)
        if self.disposition == "GAP":
            if (
                type(self.reason) is not str
                or not self.reason
                or self.minimum_primitive_count is not None
                or self.hypotheses
            ):
                raise ConnectedSynthesisError("GAP payload differs")
            return
        if self.reason is not None or not self.hypotheses:
            raise ConnectedSynthesisError("successful payload differs")
        if (
            type(self.minimum_primitive_count) is not int
            or not 1 <= self.minimum_primitive_count <= MAX_VISIBLE_PRIMITIVES
            or any(
                len(item.primitives) != self.minimum_primitive_count
                for item in self.hypotheses
            )
        ):
            raise ConnectedSynthesisError("minimum primitive count differs")
        expected_disposition = "IDENTIFIED" if len(self.hypotheses) == 1 else "AMBIGUOUS"
        if self.disposition != expected_disposition:
            raise ConnectedSynthesisError("outcome ambiguity differs")
        keys = tuple(item.geometry_key for item in self.hypotheses)
        reconstructed = {
            item.reconstructed_ink_pixels for item in self.hypotheses
        }
        if (
            len(keys) != len(set(keys))
            or keys != tuple(sorted(keys))
            or len(reconstructed) != 1
            or len(next(iter(reconstructed), ())) != self.foreground_pixel_count
        ):
            raise ConnectedSynthesisError("outcome repeats or misorders geometry")

    @property
    def candidate_pairs(self) -> tuple[tuple[int, int], ...]:
        return tuple(sorted({item.pair for item in self.hypotheses}))

    @property
    def exact_reconstruction(self) -> bool:
        return bool(self.hypotheses) and all(
            item.xor_pixel_count == 0 and item.intersection_over_union == 1.0
            for item in self.hypotheses
        )


@dataclass(frozen=True, slots=True)
class _CatalogMask:
    primitive_id: str
    equivalent_ids: tuple[str, ...]
    kind: _Kind
    ink_pixels: tuple[int, ...]
    bits: int
    boundary_yx: tuple[tuple[int, int], ...]
    endpoints_yx: tuple[tuple[int, int], ...]


def _decode_exact_mask(png_bytes: bytes) -> np.ndarray:
    try:
        with Image.open(BytesIO(png_bytes)) as image:
            if (
                image.format != "PNG"
                or getattr(image, "n_frames", 1) != 1
                or image.size != (IMAGE_SIZE, IMAGE_SIZE)
            ):
                raise ConnectedSynthesisError("issued transport must be one fixed-size PNG")
            image.load()
            gray = np.asarray(image.convert("L"), dtype=np.uint8)
    except ConnectedSynthesisError:
        raise
    except Exception as exc:
        raise ConnectedSynthesisError(f"cannot decode issued PNG: {exc}") from exc
    mask = np.ascontiguousarray(gray < 128)
    if not mask.any():
        raise ConnectedSynthesisError("issued PNG has no visible foreground")
    return mask


def _pixels_to_bits(pixels: tuple[int, ...]) -> int:
    result = 0
    for pixel in pixels:
        result |= 1 << pixel
    return result


def _normalise_yx(value: object, name: str) -> tuple[tuple[int, int], ...]:
    """Accept the sibling catalog's documented flat or ``(y, x)`` boundary."""

    if type(value) is not tuple:
        raise ConnectedSynthesisError(f"catalog {name} container differs")
    if all(type(item) is int for item in value):
        pixels = _exact_pixel_tuple(value, f"catalog {name}")
        return tuple((pixel // IMAGE_SIZE, pixel % IMAGE_SIZE) for pixel in pixels)
    result = tuple(value)
    if any(
        type(point) is not tuple
        or len(point) != 2
        or any(type(coordinate) is not int for coordinate in point)
        or any(not 0 <= coordinate < IMAGE_SIZE for coordinate in point)
        for point in result
    ):
        raise ConnectedSynthesisError(f"catalog {name} differs")
    return tuple(sorted(set(result)))


@lru_cache(maxsize=1)
def _catalog_masks() -> tuple[_CatalogMask, ...]:
    """Copy and seal the sibling's source-fixed catalog once per process."""

    rows = connected.primitive_catalog()
    if type(rows) is not tuple or not rows:
        raise ConnectedSynthesisError("primitive catalog differs")
    by_geometry: dict[tuple[str, tuple[int, ...]], list[object]] = {}
    seen_ids: set[str] = set()
    for row in rows:
        primitive_id = getattr(row, "primitive_id", None)
        kind = getattr(row, "kind", None)
        ink_pixels = getattr(row, "ink_pixels", None)
        if type(primitive_id) is not str or not primitive_id or primitive_id in seen_ids:
            raise ConnectedSynthesisError("catalog primitive id differs")
        seen_ids.add(primitive_id)
        if type(kind) is not str or kind not in ("line", "arc"):
            raise ConnectedSynthesisError("catalog primitive kind differs")
        ink = _exact_pixel_tuple(ink_pixels, "catalog primitive ink")
        if not ink:
            raise ConnectedSynthesisError("catalog primitive has no ink")
        by_geometry.setdefault((kind, ink), []).append(row)

    result: list[_CatalogMask] = []
    for (kind, ink), equivalents in by_geometry.items():
        ids = tuple(sorted(getattr(row, "primitive_id") for row in equivalents))
        # Same-kind, same-mask rows are observational aliases.  Endpoint and
        # boundary metadata must nevertheless agree before they may collapse.
        boundaries = {
            _normalise_yx(getattr(row, "boundary_pixels"), "boundary")
            for row in equivalents
        }
        endpoints = {
            _normalise_yx(getattr(row, "endpoints_yx"), "endpoints")
            for row in equivalents
        }
        if len(boundaries) != 1 or len(endpoints) != 1:
            raise ConnectedSynthesisError("equivalent catalog geometry metadata differs")
        result.append(
            _CatalogMask(
                ids[0], ids, kind, ink, _pixels_to_bits(ink),
                next(iter(boundaries)), next(iter(endpoints)),
            )
        )
    result.sort(key=lambda item: (item.primitive_id, item.kind, item.ink_pixels))
    return tuple(result)


def sealed_catalog_digest() -> str:
    """Address the exact normalized catalog snapshot used by the fitter."""

    source_sha256()
    return "sha256:" + canonical_digest(
        [
            {
                "primitive_id": item.primitive_id,
                "equivalent_primitive_ids": list(item.equivalent_ids),
                "kind": item.kind,
                "ink_pixels": list(item.ink_pixels),
                "boundary_pixels_yx": [list(point) for point in item.boundary_yx],
                "endpoints_yx": [list(point) for point in item.endpoints_yx],
            }
            for item in _catalog_masks()
        ]
    )


def _mask_boundary(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    padded = np.pad(mask, 1, constant_values=False)
    interior = np.ones_like(mask)
    for dy in range(3):
        for dx in range(3):
            interior &= padded[dy : dy + IMAGE_SIZE, dx : dx + IMAGE_SIZE]
    return tuple(tuple(map(int, point)) for point in np.argwhere(mask & ~interior))


def _graph_endpoints(paths: tuple[ordered.OrderedGraphPath, ...]) -> tuple[tuple[int, int], ...]:
    points: set[tuple[int, int]] = set()
    for path in paths:
        if not path.closed and path.start_degree == 1:
            points.add(path.pixels_yx[0])
        if not path.closed and path.end_degree == 1:
            points.add(path.pixels_yx[-1])
    return tuple(sorted(points))


def _minimum_exact_covers(target_bits: int, candidates: tuple[_CatalogMask, ...]) -> tuple[tuple[int, ...], ...]:
    """Return every minimum candidate-index cover of ``target_bits``."""

    covering: dict[int, tuple[int, ...]] = {}
    for pixel in range(PIXEL_COUNT):
        bit = 1 << pixel
        if target_bits & bit:
            covering[pixel] = tuple(
                index for index, candidate in enumerate(candidates)
                if candidate.bits & bit
            )
            if not covering[pixel]:
                return ()

    def pivot(covered: int) -> int:
        uncovered = target_bits & ~covered
        pixels: list[int] = []
        while uncovered:
            low = uncovered & -uncovered
            pixels.append(low.bit_length() - 1)
            uncovered ^= low
        return min(
            pixels,
            key=lambda pixel: (
                sum(bool(candidates[index].bits & ~covered) for index in covering[pixel]),
                pixel,
            ),
        )

    impossible = MAX_VISIBLE_PRIMITIVES + 1

    @lru_cache(maxsize=None)
    def min_additional(covered: int, slots: int) -> int:
        if covered == target_bits:
            return 0
        if slots == 0:
            return impossible
        uncovered = target_bits & ~covered
        maximum_gain = max(
            (candidate.bits & uncovered).bit_count()
            for candidate in candidates
        ) if candidates else 0
        if maximum_gain == 0 or (
            uncovered.bit_count() + maximum_gain - 1
        ) // maximum_gain > slots:
            return impossible
        pixel = pivot(covered)
        best = impossible
        for index in covering[pixel]:
            nxt = covered | candidates[index].bits
            if nxt == covered:
                continue
            remaining = min_additional(nxt, slots - 1)
            if remaining < impossible:
                best = min(best, 1 + remaining)
        return best

    minimum = min_additional(0, MAX_VISIBLE_PRIMITIVES)
    if minimum < 1 or minimum > MAX_VISIBLE_PRIMITIVES:
        return ()

    covers: set[tuple[int, ...]] = set()

    def collect(covered: int, chosen: frozenset[int]) -> None:
        if len(chosen) > minimum:
            return
        if covered == target_bits:
            if len(chosen) == minimum:
                covers.add(tuple(sorted(chosen)))
            return
        remaining_slots = minimum - len(chosen)
        if len(chosen) + min_additional(covered, remaining_slots) > minimum:
            return
        pixel = pivot(covered)
        for index in covering[pixel]:
            if index in chosen:
                continue
            nxt = covered | candidates[index].bits
            if (
                len(chosen)
                + 1
                + min_additional(nxt, remaining_slots - 1)
                <= minimum
            ):
                collect(nxt, chosen | {index})

    collect(0, frozenset())
    return tuple(sorted(covers))


def _fit_png_hypotheses_from_exact_bytes(png_bytes: bytes) -> ConnectedFitOutcome:
    """Fit all minimum exact catalog-mask programs to exact PNG bytes.

    This raw API deliberately has no target argument.  The only authority it
    uses beyond the PNG itself is the fixed renderer catalog.  In particular,
    neither ``exact_cover_target`` nor any canonical target resolver is read or
    called on this path.
    """

    source_sha256()
    ordered.source_sha256()
    if type(png_bytes) is not bytes:
        raise ConnectedSynthesisError("PNG transport must be exact bytes")
    mask = _decode_exact_mask(png_bytes)
    foreground = tuple(int(pixel) for pixel in np.flatnonzero(mask))
    target_bits = _pixels_to_bits(foreground)

    # Reuse the ordered observer's deterministic thinning and maximal graph
    # path tracer, but not its heuristic primitive classifier or target gate.
    try:
        skeleton = ordered._thin(mask)  # noqa: SLF001 - shared experimental primitive
        paths = ordered._trace_paths(skeleton)  # noqa: SLF001
    except Exception as exc:
        raise ConnectedSynthesisError(f"ordered skeleton tracing failed: {exc}") from exc
    if not paths:
        raise ConnectedSynthesisError("ordered skeleton tracer returned no paths")

    eligible = tuple(
        candidate for candidate in _catalog_masks()
        if candidate.bits & ~target_bits == 0
    )
    covers = _minimum_exact_covers(target_bits, eligible)
    boundary = _mask_boundary(mask)
    endpoints = _graph_endpoints(paths)
    foreground_count = len(foreground)
    skeleton_count = int(skeleton.sum())
    if not covers:
        return ConnectedFitOutcome(
            "GAP", "no_exact_catalog_cover_with_at_most_nine_primitives",
            paths, boundary, endpoints, foreground_count, skeleton_count, None, (),
        )

    path_pixel_sets = tuple(
        {y * IMAGE_SIZE + x for y, x in path.pixels_yx}
        for path in paths
    )
    hypotheses: list[ConnectedProgramHypothesis] = []
    for cover in covers:
        chosen: list[ChosenCatalogPrimitive] = []
        for index in cover:
            candidate = eligible[index]
            ink_set = set(candidate.ink_pixels)
            path_ids = tuple(
                path.path_id for path, path_pixels in zip(paths, path_pixel_sets, strict=True)
                if ink_set & path_pixels
            )
            chosen.append(
                ChosenCatalogPrimitive(
                    candidate.primitive_id,
                    candidate.equivalent_ids,
                    candidate.kind,
                    candidate.ink_pixels,
                    candidate.boundary_yx,
                    candidate.endpoints_yx,
                    path_ids,
                )
            )
        chosen.sort(key=lambda item: item.primitive_id)
        primitives = tuple(chosen)
        reconstructed = tuple(
            sorted({pixel for item in primitives for pixel in item.ink_pixels})
        )
        xor_count = len(set(reconstructed) ^ set(foreground))
        union_count = len(set(reconstructed) | set(foreground))
        intersection_count = len(set(reconstructed) & set(foreground))
        iou = float(intersection_count / union_count)
        hypotheses.append(
            ConnectedProgramHypothesis(
                sum(item.kind == "line" for item in primitives),
                sum(item.kind == "arc" for item in primitives),
                primitives,
                reconstructed,
                xor_count,
                iou,
            )
        )
    hypotheses.sort(key=lambda item: item.geometry_key)
    payload = tuple(hypotheses)
    return ConnectedFitOutcome(
        "IDENTIFIED" if len(payload) == 1 else "AMBIGUOUS",
        None,
        paths,
        boundary,
        endpoints,
        foreground_count,
        skeleton_count,
        len(payload[0].primitives),
        payload,
    )


def fit_authenticated_png_hypotheses(png_bytes: bytes) -> ConnectedFitOutcome:
    """Fit bytes whose provenance was authenticated by an external custody gate.

    This function performs no synthetic-issuer check and grants no authority to
    open a file, archive, or corpus.  It is the neutral pixel-only core used
    after a caller has already obtained exact bytes from a typed release
    capability.  Its only input remains an exact in-memory PNG payload.
    """

    return _fit_png_hypotheses_from_exact_bytes(png_bytes)


def fit_png_hypotheses(png_bytes: bytes) -> ConnectedFitOutcome:
    """Fit one exact PNG previously issued by the connected synthetic renderer."""

    source_sha256()
    if type(png_bytes) is not bytes:
        raise ConnectedSynthesisError("PNG transport must be exact bytes")
    try:
        connected.require_issued_connected_png(png_bytes)
    except Exception as exc:
        raise ConnectedSynthesisError(
            "PNG was not issued by the connected renderer"
        ) from exc
    return _fit_png_hypotheses_from_exact_bytes(png_bytes)

"""PNG-only ordered path-graph inversion for synthetic Bongard experiments.

Unlike the frozen 112-scalar observer, this module retains skeleton incidence
and the ordered pixels of every maximal graph path.  It fits those paths with
line and circular-arc primitives and returns a *set* of count hypotheses that
are minimum-complexity under this module's bounded sampled fit grammar and
caller-supplied tolerances.  This is heuristic engineering evidence, not a
proof of a raster-minimal explanation.  Generator action history is
intentionally not reconstructed. Full-raster singleton line/arc predicates
serve only as post-fit fail-closed checks; they do not select a target pair.
Candidate pairs are independently fitted, but the shared partial-target
resolver deliberately suppresses any singleton when a connected raster is
unresolved. Structural set/GAP safety is therefore a target-derived policy
guarantee, not independent observer evidence.

This is an experimental, synthetic-only library.  It has no corpus loader,
CLI, label reader, calibration path, or benchmark authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import math
from typing import Final, Literal

import numpy as np
from PIL import Image
from scipy import ndimage

from bongard.runtime_source_snapshot import capture_loaded_source, verify_loaded_source
from bongard import panel_action_count_synthetic_identifiability as synthetic


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)

MAX_PNG_BYTES: Final = 4 * 1024 * 1024
MAX_SIDE: Final = 512
MAX_SKELETON_PIXELS: Final = 65_536
MAX_VISIBLE_PRIMITIVES: Final = 9
MIN_ARC_SWEEP_RADIANS: Final = 0.05
_DIRECTIONS: Final = (
    (-1, 0), (-1, 1), (0, 1), (1, 1),
    (1, 0), (1, -1), (0, -1), (-1, -1),
)
_Kind = Literal["line", "arc"]


class OrderedPathInversionError(ValueError):
    """A PNG or bounded graph invariant differs."""


def source_sha256() -> str:
    return verify_loaded_source(__name__, expected_source_sha256=_LOADED_SOURCE_SHA256)


@dataclass(frozen=True, slots=True)
class OrderedGraphPath:
    path_id: int
    component_id: int
    start_degree: int
    end_degree: int
    closed: bool
    pixels_yx: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        if type(self.path_id) is not int or self.path_id < 0:
            raise OrderedPathInversionError("path id differs")
        if type(self.component_id) is not int or self.component_id < 0:
            raise OrderedPathInversionError("component id differs")
        if (
            type(self.start_degree) is not int
            or type(self.end_degree) is not int
            or not 0 <= self.start_degree <= 8
            or not 0 <= self.end_degree <= 8
        ):
            raise OrderedPathInversionError("path endpoint degree differs")
        if type(self.closed) is not bool:
            raise OrderedPathInversionError("path closure differs")
        if type(self.pixels_yx) is not tuple or len(self.pixels_yx) < 2:
            raise OrderedPathInversionError("ordered path is too short")
        if any(
            type(point) is not tuple
            or len(point) != 2
            or any(type(value) is not int for value in point)
            or any(not 0 <= value < MAX_SIDE for value in point)
            for point in self.pixels_yx
        ):
            raise OrderedPathInversionError("ordered path coordinates differ")
        if any(
            max(abs(first[0] - second[0]), abs(first[1] - second[1])) != 1
            for first, second in zip(self.pixels_yx, self.pixels_yx[1:])
        ):
            raise OrderedPathInversionError("ordered path incidence differs")
        if self.closed != (self.pixels_yx[0] == self.pixels_yx[-1]):
            raise OrderedPathInversionError("ordered path closure semantics differ")
        unique = self.pixels_yx[:-1] if self.closed else self.pixels_yx
        if len(set(unique)) != len(unique):
            raise OrderedPathInversionError("ordered path repeats a pixel")
        if self.closed and (self.start_degree != 2 or self.end_degree != 2):
            raise OrderedPathInversionError("closed path endpoint degrees differ")
        if not self.closed and (
            self.start_degree in (0, 2) or self.end_degree in (0, 2)
        ):
            raise OrderedPathInversionError("open path endpoint degrees differ")


@dataclass(frozen=True, slots=True)
class PrimitiveFit:
    path_id: int
    segment_start: int
    segment_end: int
    kind: _Kind
    rms_error: float
    complexity: int

    def __post_init__(self) -> None:
        if (
            type(self.path_id) is not int
            or type(self.segment_start) is not int
            or type(self.segment_end) is not int
            or self.path_id < 0
            or self.segment_start < 0
            or self.segment_end <= self.segment_start
        ):
            raise OrderedPathInversionError("primitive-fit indices differ")
        if type(self.kind) is not str or self.kind not in {"line", "arc"}:
            raise OrderedPathInversionError("primitive-fit kind differs")
        if (
            type(self.rms_error) is not float
            or not math.isfinite(self.rms_error)
            or self.rms_error < 0
        ):
            raise OrderedPathInversionError("primitive-fit error differs")
        if type(self.complexity) is not int or self.complexity != 1:
            raise OrderedPathInversionError("primitive-fit complexity differs")


@dataclass(frozen=True, slots=True)
class ProgramHypothesis:
    straight_count: int
    arc_count: int
    total_rms_error: float
    fits: tuple[PrimitiveFit, ...]

    def __post_init__(self) -> None:
        if (
            type(self.straight_count) is not int
            or type(self.arc_count) is not int
            or not 1 <= self.straight_count + self.arc_count <= 9
        ):
            raise OrderedPathInversionError("hypothesis count pair differs")
        if type(self.total_rms_error) is not float or not math.isfinite(self.total_rms_error):
            raise OrderedPathInversionError("hypothesis score differs")
        if (
            type(self.fits) is not tuple
            or any(type(item) is not PrimitiveFit for item in self.fits)
            or len(self.fits) != self.straight_count + self.arc_count
        ):
            raise OrderedPathInversionError("hypothesis fit inventory differs")
        if (
            sum(item.kind == "line" for item in self.fits) != self.straight_count
            or sum(item.kind == "arc" for item in self.fits) != self.arc_count
            or not math.isclose(
                self.total_rms_error,
                sum(item.rms_error for item in self.fits),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        ):
            raise OrderedPathInversionError("hypothesis fit semantics differ")
        grouped: dict[int, list[PrimitiveFit]] = {}
        for fit in self.fits:
            grouped.setdefault(fit.path_id, []).append(fit)
        if tuple(grouped) != tuple(sorted(grouped)) or any(
            rows[0].segment_start != 0
            or any(
                first.segment_end != second.segment_start
                for first, second in zip(rows, rows[1:])
            )
            for rows in grouped.values()
        ):
            raise OrderedPathInversionError("hypothesis path cover differs")

    @property
    def pair(self) -> tuple[int, int]:
        return self.straight_count, self.arc_count


@dataclass(frozen=True, slots=True)
class InversionOutcome:
    disposition: Literal["IDENTIFIED", "AMBIGUOUS", "GAP"]
    paths: tuple[OrderedGraphPath, ...]
    hypotheses: tuple[ProgramHypothesis, ...]
    reason: str | None
    foreground_pixel_count: int
    skeleton_pixel_count: int

    def __post_init__(self) -> None:
        if type(self.disposition) is not str or self.disposition not in {"IDENTIFIED", "AMBIGUOUS", "GAP"}:
            raise OrderedPathInversionError("outcome disposition differs")
        if (
            type(self.paths) is not tuple
            or any(type(item) is not OrderedGraphPath for item in self.paths)
            or type(self.hypotheses) is not tuple
            or any(type(item) is not ProgramHypothesis for item in self.hypotheses)
        ):
            raise OrderedPathInversionError("outcome containers differ")
        if (
            type(self.foreground_pixel_count) is not int
            or type(self.skeleton_pixel_count) is not int
            or self.foreground_pixel_count < 0
            or self.skeleton_pixel_count < 0
        ):
            raise OrderedPathInversionError("outcome pixel counts differ")
        path_by_id = {path.path_id: path for path in self.paths}
        if len(path_by_id) != len(self.paths) or tuple(path_by_id) != tuple(
            range(len(self.paths))
        ):
            raise OrderedPathInversionError("outcome path ids differ")
        if self.disposition == "GAP":
            if self.hypotheses or type(self.reason) is not str or not self.reason:
                raise OrderedPathInversionError("GAP payload differs")
        else:
            if not self.hypotheses or self.reason is not None:
                raise OrderedPathInversionError("identified payload differs")
            expected = "IDENTIFIED" if len({item.pair for item in self.hypotheses}) == 1 else "AMBIGUOUS"
            if self.disposition != expected:
                raise OrderedPathInversionError("outcome ambiguity differs")
            if len({item.pair for item in self.hypotheses}) != len(self.hypotheses):
                raise OrderedPathInversionError("outcome repeats a count pair")
            for hypothesis in self.hypotheses:
                if {fit.path_id for fit in hypothesis.fits} != set(path_by_id):
                    raise OrderedPathInversionError("outcome fit path ids differ")
                fits_by_path = {
                    path_id: tuple(
                        fit for fit in hypothesis.fits if fit.path_id == path_id
                    )
                    for path_id in path_by_id
                }
                if any(not fits for fits in fits_by_path.values()) or any(
                    fits[-1].segment_end
                    != len(path_by_id[path_id].pixels_yx) - 1
                    for path_id, fits in fits_by_path.items()
                ):
                    raise OrderedPathInversionError("outcome path coverage differs")

    @property
    def candidate_pairs(self) -> tuple[tuple[int, int], ...]:
        return tuple(sorted({item.pair for item in self.hypotheses}))


def _decode_png(png_bytes: bytes) -> np.ndarray:
    if type(png_bytes) is not bytes or not 0 < len(png_bytes) <= MAX_PNG_BYTES:
        raise OrderedPathInversionError("PNG byte count leaves the fixed cap")
    try:
        with Image.open(BytesIO(png_bytes)) as image:
            if image.format != "PNG" or getattr(image, "n_frames", 1) != 1:
                raise OrderedPathInversionError("input must be one PNG frame")
            if not 1 <= image.width <= MAX_SIDE or not 1 <= image.height <= MAX_SIDE:
                raise OrderedPathInversionError("PNG dimensions leave the fixed cap")
            image.load()
            gray = np.asarray(image.convert("L"), dtype=np.uint8)
    except OrderedPathInversionError:
        raise
    except Exception as exc:
        raise OrderedPathInversionError(f"cannot decode PNG: {exc}") from exc
    mask = gray < 128
    if not mask.any():
        raise OrderedPathInversionError("PNG has no visible foreground")
    return np.ascontiguousarray(mask)


def _thin(mask: np.ndarray) -> np.ndarray:
    current = np.ascontiguousarray(mask, dtype=bool)
    work = np.pad(current, 1, constant_values=False)
    for _ in range(max(work.shape) * 2):
        changed = False
        for phase in (0, 1):
            padded = np.pad(work, 1, constant_values=False)
            neighbours = (
                padded[:-2, 1:-1], padded[:-2, 2:], padded[1:-1, 2:],
                padded[2:, 2:], padded[2:, 1:-1], padded[2:, :-2],
                padded[1:-1, :-2], padded[:-2, :-2],
            )
            count = sum(item.astype(np.uint8) for item in neighbours)
            transitions = sum(
                ((~neighbours[index]) & neighbours[(index + 1) % 8]).astype(np.uint8)
                for index in range(8)
            )
            p2, _p3, p4, _p5, p6, _p7, p8, _p9 = neighbours
            gates = (
                (~(p2 & p4 & p6), ~(p4 & p6 & p8))
                if phase == 0
                else (~(p2 & p4 & p8), ~(p2 & p6 & p8))
            )
            delete = work & (count >= 2) & (count <= 6) & (transitions == 1) & gates[0] & gates[1]
            if delete.any():
                work &= ~delete
                changed = True
        if not changed:
            result = work[1:-1, 1:-1]
            if not result.any() or int(result.sum()) > MAX_SKELETON_PIXELS:
                raise OrderedPathInversionError("skeleton size leaves the fixed bounds")
            return np.ascontiguousarray(result)
    raise OrderedPathInversionError("thinning failed to converge")


def _adjacency(skeleton: np.ndarray) -> dict[tuple[int, int], tuple[tuple[int, int], ...]]:
    points = {tuple(map(int, point)) for point in np.argwhere(skeleton)}
    edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for point in points:
        for dy, dx in _DIRECTIONS:
            candidate = point[0] + dy, point[1] + dx
            if candidate in points and point < candidate:
                edges.add((point, candidate))
    # Delete the longest edge of every 8-neighbour raster triangle.  The
    # remaining two-edge stair step represents one curve without inventing a
    # degree-three junction.  Lexicographic tie-breaking is deterministic.
    for a in sorted(points):
        neighbours = sorted(
            b for b in points if b != a and max(abs(a[0] - b[0]), abs(a[1] - b[1])) == 1
        )
        for index, b in enumerate(neighbours):
            for c in neighbours[index + 1 :]:
                bc = tuple(sorted((b, c)))
                if bc not in edges:
                    continue
                triangle = (tuple(sorted((a, b))), tuple(sorted((a, c))), bc)
                lengths = [
                    (edge[0][0] - edge[1][0]) ** 2 + (edge[0][1] - edge[1][1]) ** 2
                    for edge in triangle
                ]
                longest = max(lengths)
                edge = max(edge for edge, length in zip(triangle, lengths, strict=True) if length == longest)
                edges.discard(edge)
    result_lists = {point: [] for point in points}
    for a, b in edges:
        result_lists[a].append(b)
        result_lists[b].append(a)
    return {point: tuple(sorted(result_lists[point])) for point in sorted(points)}


def _components(adjacency: dict[tuple[int, int], tuple[tuple[int, int], ...]]) -> dict[tuple[int, int], int]:
    result: dict[tuple[int, int], int] = {}
    for root in adjacency:
        if root in result:
            continue
        component = len(set(result.values()))
        stack = [root]
        result[root] = component
        while stack:
            for neighbour in adjacency[stack.pop()]:
                if neighbour not in result:
                    result[neighbour] = component
                    stack.append(neighbour)
    return result


def _trace_paths(skeleton: np.ndarray) -> tuple[OrderedGraphPath, ...]:
    adjacency = _adjacency(skeleton)
    components = _components(adjacency)
    component_points: dict[int, tuple[tuple[int, int], ...]] = {}
    for component in sorted(set(components.values())):
        component_points[component] = tuple(
            point for point in adjacency if components[point] == component
        )

    # A nonbranching component can contain a one- or two-pixel traversal
    # artifact after thinning. Recover its ordered geodesic backbone only when
    # the graph has no degree-three node. Even a one-pixel residual attached at
    # a real degree-three node is semantic uncertainty and must survive so the
    # caller returns GAP rather than silently deleting a short branch.
    backbone_by_component: dict[int, tuple[tuple[int, int], ...]] = {}
    for component, points in component_points.items():
        if any(len(adjacency[point]) >= 3 for point in points):
            continue
        endpoints = tuple(point for point in points if len(adjacency[point]) == 1)
        best_path: tuple[tuple[int, int], ...] = ()
        for root in endpoints:
            queue = [root]
            parent: dict[tuple[int, int], tuple[int, int] | None] = {root: None}
            for current in queue:
                for neighbour in adjacency[current]:
                    if neighbour not in parent:
                        parent[neighbour] = current
                        queue.append(neighbour)
            for target in endpoints:
                if target not in parent:
                    continue
                path: list[tuple[int, int]] = []
                current: tuple[int, int] | None = target
                while current is not None:
                    path.append(current)
                    current = parent[current]
                path.reverse()
                candidate = tuple(path)
                if (len(candidate), candidate) > (len(best_path), best_path):
                    best_path = candidate
        if (
            best_path
            and len(points) - len(best_path) <= 2
            and len(best_path) / len(points) >= 0.97
        ):
            backbone_by_component[component] = best_path

    edges = {
        tuple(sorted((point, neighbour)))
        for point, neighbours in adjacency.items()
        for neighbour in neighbours
    }
    visited: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    rows: list[OrderedGraphPath] = []

    for component, pixels in sorted(backbone_by_component.items()):
        rows.append(
            OrderedGraphPath(
                len(rows), component, len(adjacency[pixels[0]]),
                len(adjacency[pixels[-1]]), False, pixels,
            )
        )
        visited.update(
            tuple(sorted((first, second)))
            for first, second in zip(pixels, pixels[1:])
        )

    def trace(start: tuple[int, int], neighbour: tuple[int, int]) -> tuple[tuple[int, int], ...]:
        path = [start, neighbour]
        visited.add(tuple(sorted((start, neighbour))))
        previous, current = start, neighbour
        while len(adjacency[current]) == 2:
            candidates = [item for item in adjacency[current] if item != previous]
            if not candidates:
                break
            nxt = candidates[0]
            edge = tuple(sorted((current, nxt)))
            if edge in visited:
                break
            visited.add(edge)
            path.append(nxt)
            previous, current = current, nxt
        return tuple(path)

    nodes = tuple(point for point in adjacency if len(adjacency[point]) != 2)
    for start in nodes:
        if components[start] in backbone_by_component:
            continue
        for neighbour in adjacency[start]:
            edge = tuple(sorted((start, neighbour)))
            if edge in visited:
                continue
            pixels = trace(start, neighbour)
            rows.append(
                OrderedGraphPath(
                    len(rows), components[start], len(adjacency[pixels[0]]),
                    len(adjacency[pixels[-1]]), False, pixels,
                )
            )
    # Components with degree two everywhere are closed loops.  Start from the
    # lexicographically smallest pixel to make their traversal deterministic.
    for edge in sorted(edges - visited):
        start, neighbour = edge
        if components[start] in backbone_by_component:
            continue
        pixels = list(trace(start, neighbour))
        if pixels[-1] != start and start in adjacency[pixels[-1]]:
            visited.add(tuple(sorted((pixels[-1], start))))
            pixels.append(start)
        rows.append(OrderedGraphPath(len(rows), components[start], 2, 2, pixels[-1] == start, tuple(pixels)))
    return tuple(rows)


def _single_path_has_unexplained_foreground(
    mask: np.ndarray,
    skeleton: np.ndarray,
    paths: tuple[OrderedGraphPath, ...],
) -> bool:
    """Detect visible side ink erased before a one-path graph was recovered.

    Zhang--Suen thinning can delete a short crossbar completely, leaving the
    same degree-two centerline as a plain stroke.  For exactly one open,
    endpoint-to-endpoint recovered path, infer the ordinary stroke envelope
    from the 90th-percentile foreground-to-centerline distance.  Any ink more
    than one additional pixel outside that envelope is not explained by the
    recovered path and forces GAP.  This is a conservative bounded raster
    check, not a claim that all possible junctions are detectable.
    """

    if (
        len(paths) != 1
        or paths[0].closed
        or paths[0].start_degree != 1
        or paths[0].end_degree != 1
    ):
        return False
    distances = ndimage.distance_transform_cdt(
        ~skeleton, metric="chessboard"
    )[mask]
    if distances.ndim != 1 or len(distances) < 2:
        raise OrderedPathInversionError("foreground residual inventory differs")
    ordered_distances = np.sort(distances)
    rank = (9 * (len(ordered_distances) - 1) + 9) // 10
    envelope = int(ordered_distances[rank])
    return bool(np.any(ordered_distances > envelope + 1))


def _line_error(points_yx: tuple[tuple[int, int], ...]) -> float:
    points = np.asarray([(x, y) for y, x in points_yx], dtype=np.float64)
    centered = points - points.mean(axis=0)
    _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]
    return float(np.sqrt(np.mean(np.square(centered @ normal))))


def _circle_error(points_yx: tuple[tuple[int, int], ...]) -> tuple[float, float]:
    if len(points_yx) < 5:
        return math.inf, 0.0
    points = np.asarray([(x, y) for y, x in points_yx], dtype=np.float64)
    a = np.column_stack((2 * points[:, 0], 2 * points[:, 1], np.ones(len(points))))
    b = np.square(points[:, 0]) + np.square(points[:, 1])
    solution, _residuals, rank, _singular = np.linalg.lstsq(a, b, rcond=None)
    if rank < 3:
        return math.inf, 0.0
    cx, cy, offset = solution
    radius2 = offset + cx * cx + cy * cy
    if radius2 <= 0:
        return math.inf, 0.0
    radii = np.hypot(points[:, 0] - cx, points[:, 1] - cy)
    angles = np.unwrap(np.arctan2(points[:, 1] - cy, points[:, 0] - cx))
    return float(np.sqrt(np.mean(np.square(radii - math.sqrt(radius2))))), float(abs(angles[-1] - angles[0]))


def _curvature_sign_consistent(points_yx: tuple[tuple[int, int], ...]) -> bool:
    """Reject an arc fit that crosses a visible inflection boundary."""

    if len(points_yx) < 9:
        return True
    points = np.asarray([(x, y) for y, x in points_yx], dtype=np.float64)
    sample_count = min(13, max(7, len(points_yx) // 3))
    indices = np.rint(np.linspace(0, len(points_yx) - 1, sample_count)).astype(int)
    sampled = points[indices]
    before = sampled[1:-1] - sampled[:-2]
    after = sampled[2:] - sampled[1:-1]
    cross = before[:, 0] * after[:, 1] - before[:, 1] * after[:, 0]
    scale = np.linalg.norm(before, axis=1) * np.linalg.norm(after, axis=1)
    meaningful = cross[np.abs(cross) > np.maximum(0.08 * scale, 0.5)]
    if len(meaningful) < 2:
        return True
    positive = int(np.count_nonzero(meaningful > 0))
    negative = int(np.count_nonzero(meaningful < 0))
    return min(positive, negative) <= max(1, round(0.20 * len(meaningful)))


def _minimum_partitions(
    points: tuple[tuple[int, int], ...],
    *,
    line_tolerance: float,
    arc_tolerance: float,
) -> tuple[tuple[tuple[_Kind, int, int, float], ...], ...]:
    """Return one best cover for every globally minimal count pair.

    The state space is explicitly bounded by 64 sampled points and nine
    visible primitives.  A state is keyed by ``(line_count, arc_count)``;
    alternatives with the same pair retain the minimum-error representative,
    while alternatives with different pairs are never compared away.  This is
    important because pixels often support several equally parsimonious
    visible decompositions under the bounded sampled fit grammar.
    """

    Part = tuple[_Kind, int, int, float]
    Cover = tuple[Part, ...]
    # dp[end][pair] = (summed RMS, deterministic representative) for a cover
    # of points[0:end + 1].  Adjacent primitives share their boundary point.
    dp: list[dict[tuple[int, int], tuple[float, Cover]]] = [
        {} for _ in points
    ]
    dp[0][(0, 0)] = (0.0, ())
    for end in range(2, len(points)):
        for start in range(0, end - 1):
            if not dp[start]:
                continue
            segment = points[start : end + 1]
            fits: list[tuple[_Kind, float]] = []
            line = _line_error(segment)
            if line <= line_tolerance:
                fits.append(("line", line))
            circle, sweep = _circle_error(segment)
            if (
                circle <= arc_tolerance
                # Reject only numerically degenerate circle fits here. A raw
                # full-raster annular-sector check below decides whether a
                # singleton arc is actually licensed.
                and sweep >= MIN_ARC_SWEEP_RADIANS
            ):
                fits.append(("arc", circle))
            for (line_count, arc_count), (prior_error, prior) in sorted(dp[start].items()):
                for kind, error in fits:
                    pair = (
                        line_count + (kind == "line"),
                        arc_count + (kind == "arc"),
                    )
                    if sum(pair) > MAX_VISIBLE_PRIMITIVES:
                        continue
                    cover = prior + ((kind, start, end, float(error)),)
                    candidate = (prior_error + error, cover)
                    incumbent = dp[end].get(pair)
                    if incumbent is None or candidate < incumbent:
                        dp[end][pair] = candidate
    if not dp[-1]:
        return ()
    minimum_complexity = min(sum(pair) for pair in dp[-1])
    return tuple(
        dp[-1][pair][1]
        for pair in sorted(dp[-1])
        if sum(pair) == minimum_complexity
    )


def _fit_path(path: OrderedGraphPath, *, line_tolerance: float, arc_tolerance: float) -> tuple[tuple[PrimitiveFit, ...], ...]:
    original_points = path.pixels_yx
    if len(original_points) > 64:
        sampled_indices = tuple(
            dict.fromkeys(
                int(round(value))
                for value in np.linspace(0, len(original_points) - 1, 64)
            )
        )
    else:
        sampled_indices = tuple(range(len(original_points)))
    points = tuple(original_points[index] for index in sampled_indices)

    def primitive_fit(
        start: int, end: int, kind: _Kind, error: float
    ) -> PrimitiveFit:
        return PrimitiveFit(
            path.path_id,
            sampled_indices[start],
            sampled_indices[end],
            kind,
            float(error),
            1,
        )

    partitions = _minimum_partitions(
        points, line_tolerance=line_tolerance, arc_tolerance=arc_tolerance
    )
    return tuple(
        tuple(
            primitive_fit(start, end, kind, error)
            for kind, start, end, error in partition
        )
        for partition in partitions
    )


def invert_png(
    png_bytes: bytes,
    *,
    line_tolerance: float = 0.55,
    arc_tolerance: float = 0.70,
) -> InversionOutcome:
    """Return set-valued counts minimal under the bounded sampled fit grammar.

    The result is non-authorizing heuristic engineering evidence.  It neither
    recovers generator action history nor certifies a raster-minimal program.
    """

    source_sha256()
    try:
        synthetic.require_issued_synthetic_png(png_bytes)
    except synthetic.SyntheticIdentifiabilityError as exc:
        raise OrderedPathInversionError(str(exc)) from exc
    if type(line_tolerance) is not float or not 0.0 < line_tolerance <= 4.0:
        raise OrderedPathInversionError("line tolerance differs")
    if type(arc_tolerance) is not float or not 0.0 < arc_tolerance <= 4.0:
        raise OrderedPathInversionError("arc tolerance differs")
    mask = _decode_png(png_bytes)
    skeleton = _thin(mask)
    paths = _trace_paths(skeleton)
    if not paths:
        return InversionOutcome("GAP", (), (), "no_ordered_skeleton_paths", int(mask.sum()), int(skeleton.sum()))
    if any(path.start_degree >= 3 or path.end_degree >= 3 for path in paths):
        return InversionOutcome(
            "GAP",
            paths,
            (),
            "junction_graph_requires_global_path_cover",
            int(mask.sum()),
            int(skeleton.sum()),
        )
    if _single_path_has_unexplained_foreground(mask, skeleton, paths):
        return InversionOutcome(
            "GAP",
            paths,
            (),
            "foreground_residual_exceeds_single_path_stroke_envelope",
            int(mask.sum()),
            int(skeleton.sum()),
        )
    alternatives = [_fit_path(path, line_tolerance=line_tolerance, arc_tolerance=arc_tolerance) for path in paths]
    if any(not item for item in alternatives):
        return InversionOutcome("GAP", paths, (), "path_has_no_bounded_line_or_arc_fit", int(mask.sum()), int(skeleton.sum()))
    # Compose paths in pair-keyed dynamic-program states.  At most 55 count
    # pairs satisfy straight + arc <= 9, so this cannot grow exponentially
    # with the number of ambiguous paths.  Keep one minimum-error visible
    # program per pair under these fits/tolerances; hidden action-boundary
    # variants are not retained.
    combined: dict[tuple[int, int], tuple[float, tuple[PrimitiveFit, ...]]] = {
        (0, 0): (0.0, ())
    }
    for path_alternatives in alternatives:
        updated: dict[tuple[int, int], tuple[float, tuple[PrimitiveFit, ...]]] = {}
        for prefix_pair, (prefix_error, prefix) in sorted(combined.items()):
            for suffix in path_alternatives:
                suffix_pair = (
                    sum(fit.kind == "line" for fit in suffix),
                    sum(fit.kind == "arc" for fit in suffix),
                )
                pair = (
                    prefix_pair[0] + suffix_pair[0],
                    prefix_pair[1] + suffix_pair[1],
                )
                if sum(pair) > MAX_VISIBLE_PRIMITIVES:
                    continue
                fits = prefix + suffix
                error = prefix_error + sum(fit.rms_error for fit in suffix)
                deterministic_key = tuple(
                    (
                        fit.path_id,
                        fit.segment_start,
                        fit.segment_end,
                        fit.kind,
                        fit.rms_error,
                    )
                    for fit in fits
                )
                incumbent = updated.get(pair)
                if incumbent is None:
                    updated[pair] = (float(error), fits)
                else:
                    incumbent_key = tuple(
                        (
                            fit.path_id,
                            fit.segment_start,
                            fit.segment_end,
                            fit.kind,
                            fit.rms_error,
                        )
                        for fit in incumbent[1]
                    )
                    if (error, deterministic_key) < (incumbent[0], incumbent_key):
                        updated[pair] = (float(error), fits)
        combined = updated
    if not combined:
        return InversionOutcome("GAP", paths, (), "visible_program_exceeds_nine_primitives", int(mask.sum()), int(skeleton.sum()))
    hypotheses = tuple(
        ProgramHypothesis(pair[0], pair[1], float(error), fits)
        for pair, (error, fits) in sorted(
            combined.items(), key=lambda item: (sum(item[0]), item[0], item[1][0])
        )
    )
    minimum_complexity = min(len(hypothesis.fits) for hypothesis in hypotheses)
    minimum_hypotheses = tuple(
        hypothesis
        for hypothesis in hypotheses
        if len(hypothesis.fits) == minimum_complexity
    )
    candidate_pairs = {hypothesis.pair for hypothesis in minimum_hypotheses}
    exact_component_form = synthetic.visible_raster_component_normal_form(
        png_bytes
    )
    if (
        exact_component_form is not None
        and sum(exact_component_form.as_tuple()) == 1
        and exact_component_form.as_tuple() not in candidate_pairs
    ):
        return InversionOutcome(
            "GAP",
            paths,
            (),
            "exact_raster_normal_form_missing_from_path_hypotheses",
            int(mask.sum()),
            int(skeleton.sum()),
        )
    if minimum_complexity == 1 and len(candidate_pairs) == 1:
        pair = next(iter(candidate_pairs))
        if pair == (1, 0):
            explained = synthetic.has_bounded_exact_single_line_explanation(
                png_bytes
            )
            reason = "raw_foreground_not_explained_by_single_straight_stroke"
        elif pair == (0, 1):
            explained = synthetic.has_bounded_exact_single_arc_explanation(
                png_bytes
            )
            reason = "raw_foreground_not_explained_by_single_circular_arc"
        else:  # pragma: no cover - one fit always contributes one primitive
            explained = False
            reason = "raw_foreground_not_explained_by_single_primitive"
        if not explained:
            return InversionOutcome(
                "GAP",
                paths,
                (),
                reason,
                int(mask.sum()),
                int(skeleton.sum()),
            )
    if exact_component_form is None and len(candidate_pairs) == 1:
        return InversionOutcome(
            "GAP",
            paths,
            (),
            "unresolved_raster_component_cannot_issue_singleton",
            int(mask.sum()),
            int(skeleton.sum()),
        )
    disposition = (
        "IDENTIFIED"
        if len({hypothesis.pair for hypothesis in minimum_hypotheses}) == 1
        else "AMBIGUOUS"
    )
    return InversionOutcome(
        disposition,
        paths,
        minimum_hypotheses,
        None,
        int(mask.sum()),
        int(skeleton.sum()),
    )


__all__ = [
    "InversionOutcome",
    "OrderedGraphPath",
    "OrderedPathInversionError",
    "PrimitiveFit",
    "ProgramHypothesis",
    "invert_png",
    "source_sha256",
]

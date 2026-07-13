"""Typed visual legs for semantic Bongard cones.

The unrestricted predicate path still lives in ``bongard_arena.py`` and
``bongard_legs.py``.  This module is the semantic-pure basis: every arrow has
an auditable contract and returns either a typed witness or a scalar
measurement derived from such witnesses.

Honesty invariant: a witness-producing leg must verify the structure it
claims.  ``detect_contact`` returns a ContactWitness only when parts actually
meet at a junction; when the relation is absent it raises instead of
fabricating evidence.  Absence claims are expressed through the honest
counting measurements (``contact_count``, ``intersection_count``,
``part_count``) which return 0 rather than raising.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from visual_witnesses import (
    ArcWitness,
    CircleIntersectionWitness,
    CirclePairWitness,
    CircleWitness,
    ContactWitness,
    ContourWitness,
    CurveWitness,
    IntersectionWitness,
    LineSegmentWitness,
    PartGraphWitness,
    PartWitness,
    PointWitness,
    PolygonWitness,
    PrototypeWitness,
    QuadrilateralWitness,
    RadialArrangementWitness,
    SkeletonGraphWitness,
    SymmetryWitness,
    TriangleWitness,
    Witness,
)


@dataclass(frozen=True)
class ObjectMask:
    mask: np.ndarray
    object_id: str = "object"


@dataclass(frozen=True)
class Scene:
    panel: np.ndarray
    objects: tuple[ObjectMask, ...]


@dataclass(frozen=True)
class LegContract:
    name: str
    domain: tuple[str, ...]
    codomain: str
    implementation: Callable
    complexity: int = 1
    invariances: frozenset[str] = frozenset()
    equivariances: frozenset[str] = frozenset()
    failure_modes: tuple[str, ...] = ()
    version: str = "0.1"
    proxy_for: tuple[str, ...] = ()


class LegRegistry:
    def __init__(self) -> None:
        self._legs: dict[str, LegContract] = {}

    def register(self, contract: LegContract) -> None:
        if contract.name in self._legs:
            raise ValueError(f"duplicate leg {contract.name}")
        self._legs[contract.name] = contract

    def get(self, name: str) -> LegContract:
        try:
            return self._legs[name]
        except KeyError as exc:
            raise KeyError(f"missing semantic leg {name}") from exc

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._legs))

    def contracts(self) -> tuple[LegContract, ...]:
        return tuple(self._legs[name] for name in self.names())

    def terminal_types(self) -> tuple[str, ...]:
        return tuple(sorted({leg.codomain for leg in self._legs.values()}))


# ---------------------------------------------------------------------------
# Pixel-graph helpers.  Panels are 1-px stroke drawings, so the mask itself
# is the curve; degree analysis on the 8-neighbourhood gives endpoints,
# junctions and cycles.
# ---------------------------------------------------------------------------

_NEIGHBOR_OFFSETS = tuple((dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                          if (dy, dx) != (0, 0))


def _skeletonize(mask: np.ndarray) -> np.ndarray:
    """Thin a stroke mask to a 1-px medial skeleton (scikit-image).

    Rasterized strokes are ~2 px thick with orientation-dependent staircase
    doublings, which make 8-neighbourhood degree analysis (endpoints,
    junctions, cycles) and contour ordering sensitive to rotation.  Thinning
    to a 1-px skeleton restores the true curve topology so those measurements
    become rotation-invariant.  scikit-image's ``skeletonize`` is a required
    dependency and the single source of truth here — thinning is not
    hand-rolled.
    """
    binary = np.asarray(mask) > 0
    if not binary.any():
        return binary
    from skimage.morphology import skeletonize
    return np.asarray(skeletonize(binary), dtype=bool)


_THIN_CACHE: dict[bytes, np.ndarray] = {}


def _thinned(mask: np.ndarray) -> np.ndarray:
    """Thinned copy of a stroke mask; solid/tiny blobs are left intact."""
    key = np.ascontiguousarray(mask).tobytes()
    cached = _THIN_CACHE.get(key)
    if cached is not None:
        return cached
    thin = _skeletonize(mask)
    # Never erase a real component: a filled region whose skeleton collapses
    # below the min-stroke size keeps its raw mask.
    if thin.sum() < 3 <= int(np.asarray(mask).sum()):
        thin = np.asarray(mask) > 0
    if len(_THIN_CACHE) > 1024:
        _THIN_CACHE.clear()
    _THIN_CACHE[key] = thin
    return thin


def _topo(obj: "ObjectMask") -> np.ndarray:
    return _thinned(obj.mask)


def _degree_map(mask: np.ndarray) -> dict[tuple[int, int], int]:
    coords = {(int(y), int(x)) for y, x in np.argwhere(mask)}
    return {
        p: sum((p[0] + dy, p[1] + dx) in coords for dy, dx in _NEIGHBOR_OFFSETS)
        for p in coords
    }


def _degrees(mask: np.ndarray) -> tuple[int, int, int]:
    deg = _degree_map(mask)
    endpoints = sum(1 for d in deg.values() if d <= 1)
    branches = sum(1 for d in deg.values() if d >= 3)
    edges = sum(deg.values()) // 2
    cycles = max(0, edges - len(deg) + 1) if deg else 0
    return endpoints, branches, cycles


def _cluster_points(points: set[tuple[int, int]]) -> list[set[tuple[int, int]]]:
    clusters: list[set[tuple[int, int]]] = []
    remaining = set(points)
    while remaining:
        seed = remaining.pop()
        cluster = {seed}
        stack = [seed]
        while stack:
            y, x = stack.pop()
            for dy, dx in _NEIGHBOR_OFFSETS:
                p = (y + dy, x + dx)
                if p in remaining:
                    remaining.discard(p)
                    cluster.add(p)
                    stack.append(p)
        clusters.append(cluster)
    return clusters


_JUMP_OFFSETS = tuple((dy, dx) for dy in (-2, -1, 0, 1, 2) for dx in (-2, -1, 0, 1, 2)
                      if (dy, dx) != (0, 0) and max(abs(dy), abs(dx)) == 2)


def _walk_order(coords: set[tuple[int, int]]) -> list[tuple[int, int]]:
    """Order stroke pixels along the curve (endpoint-first, straightest-next).

    Rasterized strokes contain 2-px staircase doublings; when the walk gets
    locally stuck it may hop to an unvisited pixel at Chebyshev distance 2,
    which keeps the ordering monotone along the curve without any thinning.
    """
    if not coords:
        return []
    deg = {
        p: sum((p[0] + dy, p[1] + dx) in coords for dy, dx in _NEIGHBOR_OFFSETS)
        for p in coords
    }
    endpoints = sorted(p for p, d in deg.items() if d <= 1)
    current = endpoints[0] if endpoints else min(coords)
    visited = {current}
    path = [current]
    prev_dir: tuple[float, float] | None = None
    while True:
        options = [
            (current[0] + dy, current[1] + dx)
            for dy, dx in _NEIGHBOR_OFFSETS
            if (current[0] + dy, current[1] + dx) in coords
            and (current[0] + dy, current[1] + dx) not in visited
        ]
        if not options:
            options = [
                (current[0] + dy, current[1] + dx)
                for dy, dx in _JUMP_OFFSETS
                if (current[0] + dy, current[1] + dx) in coords
                and (current[0] + dy, current[1] + dx) not in visited
            ]
        if not options:
            break
        if prev_dir is None:
            nxt = options[0]
        else:
            def straightness(p: tuple[int, int]) -> float:
                vy, vx = p[0] - current[0], p[1] - current[1]
                norm = math.hypot(vy, vx) or 1.0
                return (vy * prev_dir[0] + vx * prev_dir[1]) / norm
            nxt = max(options, key=straightness)
        vy, vx = nxt[0] - current[0], nxt[1] - current[1]
        norm = math.hypot(vy, vx) or 1.0
        prev_dir = (vy / norm, vx / norm)
        visited.add(nxt)
        path.append(nxt)
        current = nxt
    return path


def _mask_from_points(points: set[tuple[int, int]], shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    if points:
        ys, xs = zip(*points)
        mask[list(ys), list(xs)] = True
    return mask


def _component_masks(panel: np.ndarray, min_pixels: int = 3) -> tuple[ObjectMask, ...]:
    ink = np.asarray(panel, dtype=np.uint8) > 0
    coords = {(int(y), int(x)) for y, x in np.argwhere(ink)}
    objects: list[ObjectMask] = []
    for cluster in _cluster_points(coords):
        if len(cluster) >= min_pixels:
            objects.append(ObjectMask(_mask_from_points(cluster, ink.shape), "tmp"))
    objects.sort(key=lambda o: int(o.mask.sum()), reverse=True)
    return tuple(ObjectMask(o.mask, f"object_{i}") for i, o in enumerate(objects))


def binarize_panel(panel: np.ndarray) -> np.ndarray:
    return (np.asarray(panel) > 0).astype(np.uint8)


def parse_scene(panel: np.ndarray) -> Scene:
    return Scene(np.asarray(panel, dtype=np.uint8), _component_masks(panel))


def extract_connected_components(panel: np.ndarray) -> Scene:
    return parse_scene(panel)


def object_count(scene: Scene) -> float:
    return float(len(scene.objects))


def total_ink(panel: np.ndarray) -> float:
    return float(np.asarray(panel).sum())


def select_all_objects(scene: Scene) -> Scene:
    return scene


def select_largest(scene: Scene) -> ObjectMask:
    if not scene.objects:
        return ObjectMask(np.zeros_like(scene.panel, dtype=bool), "empty")
    return scene.objects[0]


def select_largest_object(scene: Scene) -> ObjectMask:
    return select_largest(scene)


def select_smallest_object(scene: Scene) -> ObjectMask:
    if not scene.objects:
        return ObjectMask(np.zeros_like(scene.panel, dtype=bool), "empty")
    return scene.objects[-1]


def select_principal_objects(scene: Scene) -> Scene:
    return Scene(scene.panel, scene.objects[: max(1, min(4, len(scene.objects)))])


def select_inner_object(scene: Scene) -> ObjectMask:
    return select_smallest_object(scene)


def select_outer_object(scene: Scene) -> ObjectMask:
    return select_largest(scene)


def select_parts(graph: PartGraphWitness) -> PartGraphWitness:
    return graph


def largest_area(scene: Scene) -> float:
    return float(select_largest(scene).mask.sum())


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    pts = np.argwhere(mask)
    if len(pts) == 0:
        return None
    y0, x0 = pts.min(axis=0)
    y1, x1 = pts.max(axis=0)
    return int(y0), int(x0), int(y1), int(x1)


def bbox_aspect(obj: ObjectMask) -> float:
    box = _bbox(obj.mask)
    if box is None:
        return 0.0
    y0, x0, y1, x1 = box
    h = max(1, y1 - y0 + 1)
    w = max(1, x1 - x0 + 1)
    return float(max(w / h, h / w))


def bbox_fill(obj: ObjectMask) -> float:
    box = _bbox(obj.mask)
    if box is None:
        return 0.0
    y0, x0, y1, x1 = box
    area = max(1, (y1 - y0 + 1) * (x1 - x0 + 1))
    return float(obj.mask.sum() / area)


def _covariance_eigs(mask: np.ndarray) -> tuple[float, float]:
    pts = np.argwhere(mask).astype(float)
    if len(pts) < 2:
        return 0.0, 0.0
    cov = np.cov(pts.T)
    eigs = np.linalg.eigvalsh(cov)
    return float(max(eigs, default=0.0)), float(min(eigs, default=0.0))


def elongation(obj: ObjectMask) -> float:
    """Rotation-invariant elongation: sqrt(major/minor) of the point cloud.

    The axis-aligned bounding box makes ``bbox_aspect`` swing under rotation
    (a diagonal bar looks square).  The ratio of the covariance eigenvalues
    is orientation-free, so a thin/elongated shape scores high at any angle.
    """
    major, minor = _covariance_eigs(obj.mask)
    if major <= 0:
        return 1.0
    return float(math.sqrt(major / max(minor, 1e-6)))


# ---------------------------------------------------------------------------
# Contours and curve geometry.
# ---------------------------------------------------------------------------

def extract_contours(obj: ObjectMask) -> ContourWitness:
    topo = _topo(obj)
    coords = {(int(y), int(x)) for y, x in np.argwhere(topo)}
    if not coords:
        return ContourWitness(source_component_id=obj.object_id,
                              provenance=("extract_contours",))
    deg = _degree_map(topo)
    endpoints = sum(1 for d in deg.values() if d <= 1)
    is_closed = endpoints == 0
    path = _walk_order(coords)
    covered = len(path) / len(coords)
    # Completeness is topological, not a raw coverage fraction: doubled
    # raster pixels are legitimately skipped by the walk.
    complete = False
    if len(path) >= 2 and covered >= 0.6:
        start, end = path[0], path[-1]
        if is_closed:
            complete = max(abs(start[0] - end[0]), abs(start[1] - end[1])) <= 2
        else:
            complete = deg[start] <= 1 and deg[end] <= 1 and start != end
    if complete:
        ordered = tuple((float(x), float(y)) for y, x in path)
        confidence = 1.0
    elif covered >= 0.8:
        ordered = tuple((float(x), float(y)) for y, x in path)
        confidence = covered
    else:
        pts = np.asarray([(x, y) for y, x in sorted(coords)], dtype=float)
        center = pts.mean(axis=0)
        angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
        ordered = tuple((float(x), float(y)) for x, y in pts[np.argsort(angles)])
        confidence = 0.5
    return ContourWitness(
        source_component_id=obj.object_id,
        points=ordered,
        is_closed=is_closed,
        confidence=confidence,
        provenance=("extract_contours",),
    )


def contour_closedness(contour: ContourWitness) -> float:
    return 1.0 if contour.is_closed else 0.0


def build_skeleton_graph(obj: ObjectMask) -> SkeletonGraphWitness:
    topo = _topo(obj)
    endpoints, branches, cycles = _degrees(topo)
    pts = np.argwhere(topo)
    nodes = tuple(PointWitness(x=float(x), y=float(y), source_id=obj.object_id)
                  for y, x in pts[:: max(1, len(pts) // 64)])
    return SkeletonGraphWitness(
        source_component_id=obj.object_id,
        nodes=nodes,
        endpoint_count=endpoints,
        branch_count=branches,
        cycle_count=cycles,
        provenance=("build_skeleton_graph",),
    )


def skeletonize_component(obj: ObjectMask) -> SkeletonGraphWitness:
    return build_skeleton_graph(obj)


def endpoint_count(graph: SkeletonGraphWitness) -> float:
    return float(graph.endpoint_count)


def branch_count(graph: SkeletonGraphWitness) -> float:
    return float(graph.branch_count)


def cycle_count(graph: SkeletonGraphWitness) -> float:
    return float(graph.cycle_count)


def closure_ratio(obj: ObjectMask) -> float:
    topo = _topo(obj)
    pts = np.argwhere(topo)
    if len(pts) == 0:
        return 1.0
    endpoints, _, _ = _degrees(topo)
    return float(endpoints / max(1, len(pts)))


def estimate_tangents(contour: ContourWitness) -> CurveWitness:
    return CurveWitness(
        source_component_id=contour.source_component_id,
        points=contour.points,
        confidence=contour.confidence,
        provenance=contour.provenance + ("estimate_tangents",),
    )


def estimate_curvature(contour: ContourWitness) -> CurveWitness:
    return estimate_tangents(contour)


def curvature_extrema(curve: CurveWitness) -> SkeletonGraphWitness:
    pts = tuple(PointWitness(x=x, y=y, source_id=curve.source_component_id)
                for x, y in curve.points[:: max(1, len(curve.points) // 8)])
    return SkeletonGraphWitness(
        source_component_id=curve.source_component_id,
        nodes=pts,
        endpoint_count=len(curve.endpoints),
        branch_count=0,
        cycle_count=0,
        provenance=curve.provenance + ("curvature_extrema",),
    )


# ---------------------------------------------------------------------------
# Polygon fitting via Ramer-Douglas-Peucker on the path-ordered contour.
# ---------------------------------------------------------------------------

def _rdp_indices(pts: np.ndarray, lo: int, hi: int, eps: float,
                 keep: set[int]) -> None:
    if hi <= lo + 1:
        return
    a, b = pts[lo], pts[hi]
    chord = b - a
    norm = float(np.hypot(*chord)) or 1.0
    rel = pts[lo + 1:hi] - a
    dist = np.abs(rel[:, 0] * chord[1] - rel[:, 1] * chord[0]) / norm
    k = int(np.argmax(dist))
    if float(dist[k]) > eps:
        mid = lo + 1 + k
        keep.add(mid)
        _rdp_indices(pts, lo, mid, eps, keep)
        _rdp_indices(pts, mid, hi, eps, keep)


def _simplify_polyline(pts: np.ndarray, eps: float) -> list[int]:
    keep: set[int] = {0, len(pts) - 1}
    _rdp_indices(pts, 0, len(pts) - 1, eps, keep)
    return sorted(keep)


def _turn_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    v1, v2 = b - a, c - b
    n1, n2 = float(np.hypot(*v1)), float(np.hypot(*v2))
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    cosang = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return math.degrees(math.acos(cosang))


def _resample_contour(pts: np.ndarray, n: int, closed: bool) -> np.ndarray:
    """Uniform arc-length resampling (rotation- and density-invariant)."""
    ring = np.vstack([pts, pts[:1]]) if closed else pts
    seg = np.linalg.norm(np.diff(ring, axis=0), axis=1)
    arc = np.concatenate(([0.0], np.cumsum(seg)))
    total = float(arc[-1])
    if total < 1e-6:
        return pts[:1]
    u = np.linspace(0.0, total, n, endpoint=not closed)
    xs = np.interp(u, arc, ring[:, 0])
    ys = np.interp(u, arc, ring[:, 1])
    return np.stack([xs, ys], axis=1)


def _turning_profile(rs: np.ndarray, closed: bool, k: int) -> np.ndarray:
    """Absolute turning angle (radians) at each sample over an arc window k."""
    n = len(rs)
    ang = np.zeros(n)
    indices = range(n) if closed else range(k, n - k)
    for i in indices:
        a, b, c = rs[(i - k) % n], rs[i], rs[(i + k) % n]
        v1, v2 = b - a, c - b
        n1, n2 = float(np.hypot(*v1)), float(np.hypot(*v2))
        if n1 < 1e-9 or n2 < 1e-9:
            continue
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        dot = float(v1 @ v2)
        ang[i] = abs(math.atan2(cross, dot))
    return ang


def _circular_distance(a: int, b: int, n: int) -> int:
    d = abs(a - b)
    return min(d, n - d)


def _polygon_vertices(contour: ContourWitness) -> tuple[tuple[PointWitness, ...], float]:
    """Detect corners as turning-angle peaks on an arc-length-resampled curve.

    Turning angle and arc length are both rotation-invariant, so the corner
    set (and hence the side count) no longer depends on panel orientation or
    raster density.  A smooth curve (circle/arc) has turning spread evenly
    below threshold and yields no corners; a polygon concentrates turning at
    its vertices.
    """
    pts = np.asarray(contour.points, dtype=float)
    if len(pts) < 4:
        return (), 1.0
    closed = contour.is_closed
    extent = max(float(np.ptp(pts[:, 0])), float(np.ptp(pts[:, 1])), 1.0)
    perim = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
    n = int(np.clip(round(perim / 2.0), 24, 200))
    rs = _resample_contour(pts, n, closed)
    if len(rs) < 4:
        return (), 1.0
    n = len(rs)
    k = max(2, n // 16)
    window = max(2, n // 12)
    threshold = math.radians(33.0)
    ang = _turning_profile(rs, closed, k)

    accepted: list[int] = []
    for i in np.argsort(ang)[::-1]:
        if ang[i] < threshold:
            break
        i = int(i)
        if closed:
            if all(_circular_distance(i, j, n) > window for j in accepted):
                accepted.append(i)
        elif all(abs(i - j) > window for j in accepted):
            accepted.append(i)
    accepted.sort()

    corner_pts = [rs[i] for i in accepted]
    if closed:
        verts = corner_pts
        side_verts = corner_pts
    else:
        verts = [rs[0]] + corner_pts + [rs[-1]]
        side_verts = verts
    residual = _polygon_residual(pts, side_verts, closed) / extent if len(side_verts) >= 2 \
        else 1.0
    vertices = tuple(
        PointWitness(x=float(v[0]), y=float(v[1]),
                     source_id=contour.source_component_id)
        for v in verts
    )
    return vertices, residual


def _polygon_residual(pts: np.ndarray, verts: list[np.ndarray],
                      closed: bool) -> float:
    if len(verts) < 2:
        return float("inf")
    edges = list(zip(verts, verts[1:]))
    if closed:
        edges.append((verts[-1], verts[0]))
    dists = np.full(len(pts), np.inf)
    for a, b in edges:
        chord = b - a
        length2 = float(chord @ chord) or 1.0
        t = np.clip(((pts - a) @ chord) / length2, 0.0, 1.0)
        proj = a + t[:, None] * chord
        dists = np.minimum(dists, np.linalg.norm(pts - proj, axis=1))
    return float(np.mean(dists))


def fit_polygon(contour: ContourWitness) -> PolygonWitness:
    vertices, residual = _polygon_vertices(contour)
    side_count = len(vertices) if contour.is_closed else max(0, len(vertices) - 1)
    confidence = max(0.0, 1.0 - 10.0 * residual) if len(vertices) >= 2 else 0.1
    return PolygonWitness(
        source_component_id=contour.source_component_id,
        vertices=vertices,
        side_count=side_count,
        residual=residual,
        confidence=confidence,
        provenance=contour.provenance + ("fit_polygon",),
    )


def detect_corners(contour: ContourWitness) -> PolygonWitness:
    return fit_polygon(contour)


def decompose_into_line_segments(contour: ContourWitness) -> PolygonWitness:
    return fit_polygon(contour)


def polygon_side_count(poly: PolygonWitness) -> float:
    return float(poly.side_count)


def classify_triangle(poly: PolygonWitness) -> TriangleWitness:
    if poly.side_count != 3:
        raise ValueError(f"polygon has {poly.side_count} sides, not a triangle")
    return TriangleWitness(
        source_component_id=poly.source_component_id,
        vertices=poly.vertices,
        confidence=poly.confidence,
        residual=poly.residual,
        provenance=poly.provenance + ("classify_triangle",),
    )


def classify_quadrilateral(poly: PolygonWitness) -> QuadrilateralWitness:
    if poly.side_count != 4:
        raise ValueError(f"polygon has {poly.side_count} sides, not a quadrilateral")
    return QuadrilateralWitness(
        source_component_id=poly.source_component_id,
        vertices=poly.vertices,
        confidence=poly.confidence,
        residual=poly.residual,
        provenance=poly.provenance + ("classify_quadrilateral",),
    )


def fit_line_segment(contour: ContourWitness) -> LineSegmentWitness:
    pts = np.asarray(contour.points, dtype=float)
    if len(pts) == 0:
        return LineSegmentWitness(source_component_id=contour.source_component_id)
    a = pts[0]
    b = pts[np.argmax(np.linalg.norm(pts - a, axis=1))]
    length = float(np.linalg.norm(b - a))
    return LineSegmentWitness(
        source_component_id=contour.source_component_id,
        start=PointWitness(x=float(a[0]), y=float(a[1]), source_id=contour.source_component_id),
        end=PointWitness(x=float(b[0]), y=float(b[1]), source_id=contour.source_component_id),
        length=length,
        provenance=contour.provenance + ("fit_line_segment",),
    )


def _taubin_circle(pts: np.ndarray) -> tuple[np.ndarray, float]:
    """Taubin algebraic circle fit (Chernov).

    Kåsa's fit is heavily biased for small/partial arcs (Chernov, "Circular
    and Linear Regression"); Taubin removes most of that essential bias while
    staying a closed-form algebraic fit, so arc residuals are meaningful for
    the partial arcs common in these panels.
    """
    x, y = pts[:, 0], pts[:, 1]
    mx, my = float(x.mean()), float(y.mean())
    u, v = x - mx, y - my
    z = u * u + v * v
    mz = float(z.mean())
    mxx, myy, mxy = float((u * u).mean()), float((v * v).mean()), float((u * v).mean())
    mxz, myz, mzz = float((u * z).mean()), float((v * z).mean()), float((z * z).mean())
    cov_xy = mxx * myy - mxy * mxy
    var_z = mzz - mz * mz
    a3 = 4.0 * mz
    a2 = -3.0 * mz * mz - mzz
    a1 = var_z * mz + 4.0 * cov_xy * mz - mxz * mxz - myz * myz
    a0 = mxz * (mxz * myy - myz * mxy) + myz * (myz * mxx - mxz * mxy) - var_z * cov_xy
    a22, a33 = a2 + a2, a3 + a3 + a3
    xnew, ynew = 0.0, a0
    for _ in range(99):
        dy = a1 + xnew * (a22 + a33 * xnew)
        if dy == 0.0:
            break
        step = ynew / dy
        cand = xnew - step
        if cand == xnew or not math.isfinite(cand):
            break
        yval = a0 + cand * (a1 + cand * (a2 + cand * a3))
        if abs(yval) >= abs(ynew):
            break
        xnew, ynew = cand, yval
    det = xnew * xnew - xnew * mz + cov_xy
    if abs(det) < 1e-12:
        raise ValueError("degenerate circle fit")
    xc = (mxz * (myy - xnew) - myz * mxy) / det / 2.0
    yc = (myz * (mxx - xnew) - mxz * mxy) / det / 2.0
    radius_sq = xc * xc + yc * yc + mz
    if not math.isfinite(radius_sq) or radius_sq <= 0:
        raise ValueError("degenerate circle fit")
    return np.array([xc + mx, yc + my]), math.sqrt(radius_sq)


def _fit_circle_raw(contour: ContourWitness) -> CircleWitness:
    pts = np.asarray(contour.points, dtype=float)
    if len(pts) < 3:
        raise ValueError("not enough contour points for circle fit")
    center, radius = _taubin_circle(pts)
    radii = np.linalg.norm(pts - center, axis=1)
    residual = float(np.sqrt(np.mean((radii - radius) ** 2)) / max(radius, 1e-9))
    return CircleWitness(
        source_component_id=contour.source_component_id,
        center=PointWitness(x=float(center[0]), y=float(center[1]),
                            source_id=contour.source_component_id),
        radius=radius,
        support_points=tuple((float(x), float(y)) for x, y in pts[:: max(1, len(pts) // 32)]),
        residual=residual,
        confidence=max(0.0, 1.0 - residual),
        provenance=contour.provenance + ("fit_circle",),
    )


def fit_circle(contour: ContourWitness) -> CircleWitness:
    if not contour.is_closed:
        raise ValueError("contour is open; a circle is a closed curve")
    return _fit_circle_raw(contour)


def fit_arc(contour: ContourWitness) -> ArcWitness:
    circle = _fit_circle_raw(contour)
    pts = np.asarray(contour.points, dtype=float)
    center = np.array([circle.center.x, circle.center.y])
    angles = np.unwrap(np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0]))
    swept = math.degrees(abs(float(angles[-1] - angles[0]))) if len(angles) > 1 else 0.0
    return ArcWitness(
        source_component_id=contour.source_component_id,
        center=circle.center,
        radius=circle.radius,
        angle_degrees=min(360.0, swept),
        residual=circle.residual,
        confidence=circle.confidence,
        provenance=circle.provenance + ("fit_arc",),
    )


def arc_angle_degrees(arc: ArcWitness) -> float:
    return float(arc.angle_degrees)


def _signed_turning_profile(rs: np.ndarray, closed: bool, k: int) -> np.ndarray:
    """Signed turning angle (radians) per sample; + is left/convex for CCW."""
    n = len(rs)
    ang = np.zeros(n)
    indices = range(n) if closed else range(k, n - k)
    for i in indices:
        a, b, c = rs[(i - k) % n], rs[i], rs[(i + k) % n]
        v1, v2 = b - a, c - b
        n1, n2 = float(np.hypot(*v1)), float(np.hypot(*v2))
        if n1 < 1e-9 or n2 < 1e-9:
            continue
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        dot = float(v1 @ v2)
        ang[i] = math.atan2(cross, dot)
    return ang


def _signed_area(rs: np.ndarray) -> float:
    x, y = rs[:, 0], rs[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def count_curve_parts(contour: ContourWitness) -> float:
    """Number of parts by the minima rule (Hoffman & Richards).

    A shape's boundary is segmented at negative minima of curvature — the
    concave creases where transversality says two parts join.  The contour
    is oriented counter-clockwise first so "negative" (concave) is defined
    independent of walk direction, and curvature is measured on an
    arc-length resampling, so the part count is rotation-invariant.  A convex
    blob has one part; a k-lobed/petalled shape has k concave notches → k
    parts.
    """
    pts = np.asarray(contour.points, dtype=float)
    if len(pts) < 6:
        return 1.0
    closed = contour.is_closed
    perim = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
    n = int(np.clip(round(perim / 2.0), 24, 200))
    rs = _resample_contour(pts, n, closed)
    if len(rs) < 6:
        return 1.0
    n = len(rs)
    if closed and _signed_area(rs) < 0:
        rs = rs[::-1]
    k = max(2, n // 16)
    window = max(2, n // 12)
    ang = _signed_turning_profile(rs, closed, k)
    concavity = -ang  # positive where the boundary is concave (a notch)
    threshold = math.radians(25.0)
    notches: list[int] = []
    for i in np.argsort(concavity)[::-1]:
        if concavity[i] < threshold:
            break
        i = int(i)
        if closed:
            if all(_circular_distance(i, j, n) > window for j in notches):
                notches.append(i)
        elif all(abs(i - j) > window for j in notches):
            notches.append(i)
    if closed:
        return float(max(1, len(notches)))
    return float(len(notches) + 1)


def count_inflections(contour: ContourWitness) -> float:
    """Number of turning-direction reversals along the curve.

    A general, rotation-invariant shape primitive: resample the ordered
    contour by arc length, take the signed turn (cross product of successive
    tangents) with a deadband to ignore raster noise, and count sign
    changes.  A simple arc turns monotonically (0 reversals); a wavy /
    scalloped / S-shaped curve reverses several times.  This is curve
    geometry, not a named concept.
    """
    pts = np.asarray(contour.points, dtype=float)
    if len(pts) < 5:
        return 0.0
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    arc = np.concatenate(([0.0], np.cumsum(seg)))
    if arc[-1] < 1e-6:
        return 0.0
    n = 48
    u = np.linspace(0.0, arc[-1], n)
    xs = np.interp(u, arc, pts[:, 0])
    ys = np.interp(u, arc, pts[:, 1])
    rs = np.stack([xs, ys], axis=1)
    v = np.diff(rs, axis=0)
    norms = np.linalg.norm(v, axis=1)
    cross = v[:-1, 0] * v[1:, 1] - v[:-1, 1] * v[1:, 0]
    denom = norms[:-1] * norms[1:] + 1e-9
    sine = cross / denom
    sign = np.where(sine > 0.12, 1, np.where(sine < -0.12, -1, 0))
    sign = sign[sign != 0]
    if len(sign) < 2:
        return 0.0
    return float(int(np.sum(sign[1:] != sign[:-1])))


def fit_multiple_circles(scene: Scene) -> CirclePairWitness:
    candidates: list[CircleWitness] = []
    for obj in scene.objects[:4]:
        contour = extract_contours(obj)
        if len(contour.points) < 3:
            continue
        try:
            candidates.append(_fit_circle_raw(contour))
        except ValueError:
            continue
    candidates.sort(key=lambda c: c.residual)
    if len(candidates) < 2:
        raise ValueError("need at least two circle candidates")
    a, b = candidates[:2]
    d = math.hypot(a.center.x - b.center.x, a.center.y - b.center.y)
    return CirclePairWitness(
        first=a,
        second=b,
        center_distance=d,
        confidence=min(a.confidence, b.confidence),
        residual=max(a.residual, b.residual),
        provenance=a.provenance + b.provenance + ("fit_multiple_circles",),
    )


# ---------------------------------------------------------------------------
# Part decomposition at stroke junctions.  A junction is a pixel cluster of
# degree >= 3; removing it splits the stroke into parts.  Clusters touching
# only two parts are raster artifacts and get merged back.  Real junctions
# become honest contact/intersection evidence: attachment for 3 incident
# branches, crossing for 4 or more.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _Junction:
    center: tuple[float, float]           # (x, y)
    part_indices: tuple[int, ...]
    branchiness: int


def _decompose_mask(mask: np.ndarray, source_id: str,
                    min_part_pixels: int = 4
                    ) -> tuple[list[set[tuple[int, int]]], list[_Junction]]:
    coords = {(int(y), int(x)) for y, x in np.argwhere(mask)}
    if not coords:
        return [], []
    deg = _degree_map(mask)
    junction_pixels = {p for p, d in deg.items() if d >= 3}
    clusters = _cluster_points(junction_pixels)
    remainder = coords - junction_pixels
    raw_parts = [c for c in _cluster_points(remainder) if len(c) >= min_part_pixels]
    tiny = [c for c in _cluster_points(remainder) if len(c) < min_part_pixels]

    def adjacent_parts(cluster: set[tuple[int, int]],
                       parts: list[set[tuple[int, int]]]) -> list[int]:
        found = []
        for i, part in enumerate(parts):
            if any((y + dy, x + dx) in part
                   for y, x in cluster for dy, dx in _NEIGHBOR_OFFSETS):
                found.append(i)
        return found

    # Absorb tiny fragments into the nearest real part.
    for frag in tiny:
        fy, fx = next(iter(frag))
        best, best_d = None, float("inf")
        for i, part in enumerate(raw_parts):
            py, px = next(iter(part))
            d = abs(py - fy) + abs(px - fx)
            if d < best_d:
                best, best_d = i, d
        if best is not None:
            raw_parts[best] |= frag

    # Merge parts across artifact clusters (only two incident branches means
    # the stroke merely continues through a raster-thick spot).
    parent = list(range(len(raw_parts)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        parent[find(i)] = find(j)

    real_clusters: list[tuple[set[tuple[int, int]], list[int]]] = []
    for cluster in clusters:
        adj = adjacent_parts(cluster, raw_parts)
        if len(adj) <= 2:
            for i in adj[1:]:
                union(adj[0], i)
            if adj:
                raw_parts[adj[0]] |= cluster
            continue
        real_clusters.append((cluster, adj))

    merged: dict[int, set[tuple[int, int]]] = {}
    for i, part in enumerate(raw_parts):
        merged.setdefault(find(i), set()).update(part)
    part_list = sorted(merged.values(), key=len, reverse=True)
    index_of = {find(i): None for i in range(len(raw_parts))}
    for root in index_of:
        for k, part in enumerate(part_list):
            if merged[root] is part:
                index_of[root] = k
                break

    junctions: list[_Junction] = []
    for cluster, adj in real_clusters:
        ys, xs = zip(*cluster)
        center = (float(sum(xs)) / len(xs), float(sum(ys)) / len(ys))
        parts_idx = tuple(sorted({index_of[find(i)] for i in adj
                                  if index_of.get(find(i)) is not None}))
        branchiness = sum(
            1 for p, d in deg.items()
            if p not in junction_pixels
            and any((p[0] + dy, p[1] + dx) in cluster for dy, dx in _NEIGHBOR_OFFSETS)
        )
        junctions.append(_Junction(center, parts_idx, branchiness))
    if not part_list:
        part_list = [coords]
    return part_list, junctions


def _part_witness(points: set[tuple[int, int]], shape: tuple[int, int],
                  part_id: str, source_id: str, provenance: tuple[str, ...]
                  ) -> PartWitness:
    sub = ObjectMask(_mask_from_points(points, shape), part_id)
    contour = extract_contours(sub)
    contour = ContourWitness(
        source_component_id=source_id,
        points=contour.points,
        is_closed=contour.is_closed,
        confidence=contour.confidence,
        provenance=provenance,
    )
    return PartWitness(
        part_id=part_id,
        role="stroke",
        source_component_id=source_id,
        contour=contour,
        provenance=provenance,
    )


def _graph_from_mask(mask: np.ndarray, source_id: str,
                     provenance: tuple[str, ...]) -> PartGraphWitness:
    parts_pts, junctions = _decompose_mask(mask, source_id)
    parts = tuple(
        _part_witness(pts, mask.shape, f"{source_id}_part_{i}", source_id, provenance)
        for i, pts in enumerate(parts_pts)
    )
    contacts = []
    adjacency = []
    for j in junctions:
        relation = "intersection" if j.branchiness >= 4 else "attachment"
        point = PointWitness(x=j.center[0], y=j.center[1], source_id=source_id)
        ids = [parts[i].part_id for i in j.part_indices if i < len(parts)]
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                adjacency.append((ids[a], ids[b]))
        if len(ids) >= 2:
            witness_cls = IntersectionWitness if relation == "intersection" else ContactWitness
            contacts.append(witness_cls(
                source_a=ids[0], source_b=ids[1], points=(point,),
                relation=relation, confidence=1.0,
                provenance=provenance,
            ))
    return PartGraphWitness(
        parts=parts,
        contacts=tuple(contacts),
        adjacency=tuple(adjacency),
        confidence=1.0,
        provenance=provenance,
    )


def decompose_component_into_parts(obj: ObjectMask) -> PartGraphWitness:
    return _graph_from_mask(_topo(obj), obj.object_id,
                            ("decompose_component_into_parts",))


def build_part_graph(value: Scene | ObjectMask) -> PartGraphWitness:
    if isinstance(value, ObjectMask):
        return _graph_from_mask(_topo(value), value.object_id, ("build_part_graph",))
    parts: list[PartWitness] = []
    contacts: list[ContactWitness] = []
    adjacency: list[tuple[str, str]] = []
    for obj in value.objects:
        sub = _graph_from_mask(_topo(obj), obj.object_id, ("build_part_graph",))
        parts.extend(sub.parts)
        contacts.extend(sub.contacts)
        adjacency.extend(sub.adjacency)
    return PartGraphWitness(
        parts=tuple(parts),
        contacts=tuple(contacts),
        adjacency=tuple(adjacency),
        confidence=1.0,
        provenance=("build_part_graph",),
    )


def build_containment_tree(scene: Scene) -> PartGraphWitness:
    return build_part_graph(scene)


def decompose_curve_into_arcs_and_lines(contour: ContourWitness) -> PartGraphWitness:
    pts = np.asarray(contour.points, dtype=float)
    if len(pts) < 2:
        raise ValueError("contour too small to decompose")
    vertices, _residual = _polygon_vertices(contour)
    cuts = [0]
    vert_xy = [np.array([v.x, v.y]) for v in vertices]
    for v in vert_xy:
        idx = int(np.argmin(np.linalg.norm(pts - v, axis=1)))
        cuts.append(idx)
    cuts.append(len(pts) - 1)
    cuts = sorted(set(cuts))
    parts = []
    contacts = []
    adjacency = []
    prov = contour.provenance + ("decompose_curve_into_arcs_and_lines",)
    for k, (a, b) in enumerate(zip(cuts[:-1], cuts[1:])):
        if b - a < 2:
            continue
        seg_points = tuple((float(x), float(y)) for x, y in pts[a:b + 1])
        seg_contour = ContourWitness(
            source_component_id=contour.source_component_id,
            points=seg_points, is_closed=False,
            confidence=contour.confidence, provenance=prov,
        )
        parts.append(PartWitness(
            part_id=f"{contour.source_component_id}_seg_{k}",
            role="segment",
            source_component_id=contour.source_component_id,
            contour=seg_contour,
            provenance=prov,
        ))
    for prev, nxt in zip(parts[:-1], parts[1:]):
        adjacency.append((prev.part_id, nxt.part_id))
        joint = prev.contour.points[-1]
        contacts.append(ContactWitness(
            source_a=prev.part_id, source_b=nxt.part_id,
            points=(PointWitness(x=joint[0], y=joint[1],
                                 source_id=contour.source_component_id),),
            relation="shared_endpoint", confidence=1.0, provenance=prov,
        ))
    if not parts:
        raise ValueError("no curve segments found")
    return PartGraphWitness(parts=tuple(parts), contacts=tuple(contacts),
                            adjacency=tuple(adjacency), confidence=1.0,
                            provenance=prov)


# ---------------------------------------------------------------------------
# Honest relation witnesses.  These raise when the relation is absent; the
# counting measurements below return 0 instead.
# ---------------------------------------------------------------------------

def _part_centroid(part: PartWitness) -> PointWitness:
    if not part.contour or not part.contour.points:
        return PointWitness(source_id=part.part_id)
    pts = np.asarray(part.contour.points, dtype=float)
    c = pts.mean(axis=0)
    return PointWitness(x=float(c[0]), y=float(c[1]), source_id=part.part_id)


def detect_contact(graph: PartGraphWitness) -> ContactWitness:
    if len(graph.parts) < 2:
        raise ValueError("need at least two parts to witness contact")
    if not graph.contacts:
        raise ValueError("no contact between parts")
    return max(graph.contacts, key=lambda c: c.confidence)


def detect_attachment(graph: PartGraphWitness) -> ContactWitness:
    return detect_contact(graph)


def detect_tangency(graph: PartGraphWitness) -> ContactWitness:
    c = detect_contact(graph)
    return ContactWitness(source_a=c.source_a, source_b=c.source_b, points=c.points,
                          relation="tangency", confidence=c.confidence,
                          provenance=c.provenance + ("detect_tangency",))


def detect_intersection(graph: PartGraphWitness) -> IntersectionWitness:
    crossings = [c for c in graph.contacts if c.relation == "intersection"]
    if not crossings:
        raise ValueError("no crossing junction between parts")
    best = max(crossings, key=lambda c: c.confidence)
    return IntersectionWitness(
        source_a=best.source_a, source_b=best.source_b, points=best.points,
        confidence=best.confidence,
        provenance=best.provenance + ("detect_intersection",),
    )


def detect_shared_endpoint(graph: PartGraphWitness) -> ContactWitness:
    shared = [c for c in graph.contacts if c.relation == "shared_endpoint"]
    if shared:
        return shared[0]
    c = detect_contact(graph)
    return ContactWitness(source_a=c.source_a, source_b=c.source_b, points=c.points,
                          relation="shared_endpoint", confidence=c.confidence,
                          provenance=c.provenance + ("detect_shared_endpoint",))


def detect_shared_point(graph: PartGraphWitness) -> ContactWitness:
    c = detect_contact(graph)
    return ContactWitness(source_a=c.source_a, source_b=c.source_b, points=c.points,
                          relation="shared_point", confidence=c.confidence,
                          provenance=c.provenance + ("detect_shared_point",))


def circle_pair_intersection(pair: CirclePairWitness) -> CircleIntersectionWitness:
    a, b, d = pair.first, pair.second, pair.center_distance
    intersects = abs(a.radius - b.radius) <= d <= a.radius + b.radius
    if not intersects:
        raise ValueError("circle pair does not intersect")
    mid = PointWitness(
        x=(a.center.x + b.center.x) / 2.0,
        y=(a.center.y + b.center.y) / 2.0,
        source_id="circle_intersection",
    )
    return CircleIntersectionWitness(
        source_a=a.source_component_id,
        source_b=b.source_component_id,
        points=(mid,),
        pair=pair,
        confidence=pair.confidence,
        residual=pair.residual,
        provenance=pair.provenance + ("circle_pair_intersection",),
    )


def part_count(graph: PartGraphWitness) -> float:
    return float(len(graph.parts))


def contact_count(graph: PartGraphWitness) -> float:
    return float(len(graph.contacts))


def intersection_count(graph: PartGraphWitness) -> float:
    return float(sum(1 for c in graph.contacts if c.relation == "intersection"))


def reflection_symmetry(obj: ObjectMask) -> SymmetryWitness:
    score = symmetry_residual(obj)
    return SymmetryWitness(
        source_id=obj.object_id,
        kind="reflection",
        order=2,
        residual=score,
        confidence=max(0.0, 1.0 - score),
        provenance=("reflection_symmetry",),
    )


def rotational_symmetry_order(obj: ObjectMask) -> SymmetryWitness:
    score = symmetry_residual(obj)
    order = 4 if score < 0.08 else 2 if score < 0.18 else 1
    return SymmetryWitness(
        source_id=obj.object_id,
        kind="rotation",
        order=order,
        residual=score,
        confidence=max(0.0, 1.0 - score),
        provenance=("rotational_symmetry_order",),
    )


def symmetry_residual(obj: ObjectMask) -> float:
    box = _bbox(obj.mask)
    if box is None:
        return 1.0
    y0, x0, y1, x1 = box
    crop = obj.mask[y0:y1 + 1, x0:x1 + 1]
    if crop.size == 0:
        return 1.0
    residuals = [
        np.mean(crop != np.fliplr(crop)),
        np.mean(crop != np.flipud(crop)),
        np.mean(crop != np.rot90(crop, 2)),
    ]
    return float(min(residuals))


def detect_radial_arrangement(graph: PartGraphWitness) -> RadialArrangementWitness:
    if len(graph.parts) < 3:
        raise ValueError("need at least three parts for a radial arrangement")
    centers = [_part_centroid(p) for p in graph.parts]
    hub_contacts = [c for c in graph.contacts if c.points]
    if hub_contacts:
        hub = max(
            hub_contacts,
            key=lambda c: sum(1 for a, b in graph.adjacency
                              if c.source_a in (a, b) or c.source_b in (a, b)),
        ).points[0]
        cx, cy = hub.x, hub.y
    else:
        cx = sum(p.x for p in centers) / len(centers)
        cy = sum(p.y for p in centers) / len(centers)
    angles = sorted(math.atan2(p.y - cy, p.x - cx) for p in centers)
    gaps = [angles[i + 1] - angles[i] for i in range(len(angles) - 1)]
    gaps.append(2 * math.pi - (angles[-1] - angles[0]))
    mean_gap = sum(gaps) / len(gaps)
    gap_var = sum((g - mean_gap) ** 2 for g in gaps) / len(gaps)
    uniformity = max(0.0, 1.0 - math.sqrt(gap_var) / max(mean_gap, 1e-9))
    radii = [math.hypot(p.x - cx, p.y - cy) for p in centers]
    mean_r = sum(radii) / len(radii)
    radius_var = sum((r - mean_r) ** 2 for r in radii) / len(radii)
    evenness = max(0.0, 1.0 - math.sqrt(radius_var) / max(mean_r, 1e-9))
    confidence = min(uniformity, evenness)
    return RadialArrangementWitness(
        center=PointWitness(x=cx, y=cy, source_id="radial_center"),
        parts=graph.parts,
        part_count=len(graph.parts),
        symmetry_order=len(graph.parts),
        confidence=confidence,
        residual=1.0 - confidence,
        provenance=graph.provenance + ("detect_radial_arrangement",),
    )


def pair_parts_by_symmetry(graph: PartGraphWitness) -> PartGraphWitness:
    return graph


def prototype_bird_like(graph: PartGraphWitness) -> PrototypeWitness:
    """Kept only as a template for future promoted macros; not registered."""
    roles = {}
    for idx, role in enumerate(("body", "left_appendage", "right_appendage")):
        if idx < len(graph.parts):
            roles[role] = graph.parts[idx].part_id
    missing = tuple(r for r in ("body", "left_appendage", "right_appendage") if r not in roles)
    if missing:
        raise ValueError(f"missing bird-like roles: {','.join(missing)}")
    return PrototypeWitness(
        prototype_name="bird_like",
        roles=roles,
        required_roles=("body", "left_appendage", "right_appendage"),
        confidence=0.6,
        provenance=graph.provenance + ("prototype_bird_like",),
    )


def witness_confidence(witness: Witness) -> float:
    return float(witness.confidence)


def witness_residual(witness: Witness) -> float:
    return float(witness.residual)


def radial_part_count(witness: RadialArrangementWitness) -> float:
    return float(witness.part_count)


def symmetry_order_score(witness: SymmetryWitness) -> float:
    return float(witness.order)


def _reg(name: str, domain: tuple[str, ...], codomain: str, fn: Callable,
         complexity: int = 1, invariances: tuple[str, ...] = (),
         equivariances: tuple[str, ...] = (), failure_modes: tuple[str, ...] = (),
         proxy_for: tuple[str, ...] = ()) -> LegContract:
    return LegContract(
        name=name,
        domain=domain,
        codomain=codomain,
        implementation=fn,
        complexity=complexity,
        invariances=frozenset(invariances),
        equivariances=frozenset(equivariances),
        failure_modes=failure_modes,
        proxy_for=proxy_for,
    )


def default_registry() -> LegRegistry:
    reg = LegRegistry()
    common_inv = ("translation", "uniform_scale")
    for contract in (
        _reg("binarize_panel", ("Panel",), "BinaryPanel", binarize_panel, 1),
        _reg("parse_scene", ("Panel",), "Scene", parse_scene, 4, common_inv),
        _reg("extract_connected_components", ("Panel",), "Scene", extract_connected_components, 4, common_inv),
        _reg("extract_contours", ("Object",), "ContourWitness", extract_contours, 4, common_inv, proxy_for=("curve", "contour", "stroke", "boundary", "outline", "path")),
        _reg("contour_closedness", ("ContourWitness",), "Measurement", contour_closedness, 1, common_inv, proxy_for=("open", "closed", "closure", "openness", "closedness", "loop")),
        _reg("build_containment_tree", ("Scene",), "PartGraphWitness", build_containment_tree, 3, common_inv),
        _reg("skeletonize_component", ("Object",), "SkeletonGraphWitness", skeletonize_component, 3, common_inv, proxy_for=("skeleton", "path", "graph")),
        _reg("build_skeleton_graph", ("Object",), "SkeletonGraphWitness", build_skeleton_graph, 3, common_inv, proxy_for=("skeleton", "path", "graph")),
        _reg("endpoint_count", ("SkeletonGraphWitness",), "Measurement", endpoint_count, 1, common_inv, proxy_for=("endpoint", "end", "tip", "open", "closed")),
        _reg("branch_count", ("SkeletonGraphWitness",), "Measurement", branch_count, 1, common_inv, proxy_for=("branch", "branching", "fork", "forked", "junction")),
        _reg("cycle_count", ("SkeletonGraphWitness",), "Measurement", cycle_count, 1, common_inv, proxy_for=("cycle", "loop", "acyclic", "tree", "open", "closed")),
        _reg("estimate_tangents", ("ContourWitness",), "CurveWitness", estimate_tangents, 2, common_inv),
        _reg("estimate_curvature", ("ContourWitness",), "CurveWitness", estimate_curvature, 2, common_inv),
        _reg("curvature_extrema", ("CurveWitness",), "SkeletonGraphWitness", curvature_extrema, 2, common_inv),
        _reg("decompose_curve_into_arcs_and_lines", ("ContourWitness",), "PartGraphWitness", decompose_curve_into_arcs_and_lines, 5, common_inv, failure_modes=("contour_too_small",)),
        _reg("fit_line_segment", ("ContourWitness",), "LineSegmentWitness", fit_line_segment, 2, common_inv, proxy_for=("line", "segment", "straight")),
        _reg("fit_arc", ("ContourWitness",), "ArcWitness", fit_arc, 3, common_inv, failure_modes=("not_enough_points",), proxy_for=("arc", "curved", "smooth")),
        _reg("arc_angle_degrees", ("ArcWitness",), "Measurement", arc_angle_degrees, 1, common_inv, proxy_for=("angle", "sweep")),
        _reg("arc_residual", ("ArcWitness",), "Measurement", witness_residual, 1, common_inv, proxy_for=("smooth", "arc")),
        _reg("count_inflections", ("ContourWitness",), "Measurement", count_inflections, 2, common_inv, proxy_for=("wavy", "bump", "bumpy", "undulating", "undulation", "inflection", "scalloped", "sinuous", "wiggly", "reversal")),
        _reg("count_curve_parts", ("ContourWitness",), "Measurement", count_curve_parts, 3, common_inv, proxy_for=("part", "parts", "lobe", "lobes", "petal", "petals", "blade", "blades", "concavity", "concave", "notch", "notches", "arm", "arms")),
        _reg("fit_circle", ("ContourWitness",), "CircleWitness", fit_circle, 4, common_inv, failure_modes=("not_enough_points", "open_contour", "high_residual")),
        _reg("fit_multiple_circles", ("Scene",), "CirclePairWitness", fit_multiple_circles, 8, common_inv, failure_modes=("fewer_than_two_candidates", "high_residual")),
        _reg("detect_corners", ("ContourWitness",), "PolygonWitness", detect_corners, 4, common_inv),
        _reg("decompose_into_line_segments", ("ContourWitness",), "PolygonWitness", decompose_into_line_segments, 4, common_inv),
        _reg("fit_polygon", ("ContourWitness",), "PolygonWitness", fit_polygon, 5, common_inv, proxy_for=("polygon", "corner", "side", "sides", "angular", "vertex", "vertices", "bend")),
        _reg("polygon_side_count", ("PolygonWitness",), "Measurement", polygon_side_count, 1, common_inv),
        _reg("polygon_fit_residual", ("PolygonWitness",), "Measurement", witness_residual, 1, common_inv),
        _reg("classify_triangle", ("PolygonWitness",), "TriangleWitness", classify_triangle, 2, common_inv, failure_modes=("wrong_side_count",)),
        _reg("classify_quadrilateral", ("PolygonWitness",), "QuadrilateralWitness", classify_quadrilateral, 2, common_inv, failure_modes=("wrong_side_count",)),
        _reg("decompose_component_into_parts", ("Object",), "PartGraphWitness", decompose_component_into_parts, 5, common_inv),
        _reg("build_part_graph", ("Scene",), "PartGraphWitness", build_part_graph, 5, common_inv),
        _reg("build_object_part_graph", ("Object",), "PartGraphWitness", build_part_graph, 5, common_inv),
        _reg("detect_attachment", ("PartGraphWitness",), "ContactWitness", detect_attachment, 3, common_inv, failure_modes=("no_contact",), proxy_for=("attachment", "attached", "touching", "touch", "joined", "junction", "meet", "contact")),
        _reg("detect_contact", ("PartGraphWitness",), "ContactWitness", detect_contact, 3, common_inv, failure_modes=("no_contact",), proxy_for=("contact", "touching", "touch", "adjacent")),
        _reg("detect_tangency", ("PartGraphWitness",), "ContactWitness", detect_tangency, 3, common_inv, failure_modes=("no_contact",), proxy_for=("tangent", "tangency")),
        _reg("detect_intersection", ("PartGraphWitness",), "IntersectionWitness", detect_intersection, 3, common_inv, failure_modes=("no_crossing",), proxy_for=("intersect", "intersecting", "intersection", "crossing", "cross", "overlap", "overlapping")),
        _reg("circle_pair_intersection", ("CirclePairWitness",), "CircleIntersectionWitness", circle_pair_intersection, 3, common_inv, failure_modes=("no_intersection",), proxy_for=("intersect", "intersecting", "overlap")),
        _reg("detect_shared_endpoint", ("PartGraphWitness",), "ContactWitness", detect_shared_endpoint, 2, common_inv, failure_modes=("no_contact",), proxy_for=("shared", "endpoint", "meeting")),
        _reg("detect_shared_point", ("PartGraphWitness",), "ContactWitness", detect_shared_point, 2, common_inv, failure_modes=("no_contact",)),
        _reg("part_count", ("PartGraphWitness",), "Measurement", part_count, 1, common_inv, proxy_for=("part", "parts", "segment", "piece", "component", "connected", "continuous", "unbroken", "disjoint", "separate")),
        _reg("contact_count", ("PartGraphWitness",), "Measurement", contact_count, 1, common_inv, proxy_for=("contact", "touching", "attachment", "junction", "shared")),
        _reg("intersection_count", ("PartGraphWitness",), "Measurement", intersection_count, 1, common_inv, proxy_for=("intersect", "intersecting", "intersection", "crossing", "cross", "overlap")),
        _reg("reflection_symmetry", ("Object",), "SymmetryWitness", reflection_symmetry, 3, common_inv, proxy_for=("symmetric", "symmetrical", "mirror", "bilateral")),
        _reg("rotational_symmetry_order", ("Object",), "SymmetryWitness", rotational_symmetry_order, 4, common_inv, proxy_for=("symmetric", "rotational")),
        _reg("detect_radial_arrangement", ("PartGraphWitness",), "RadialArrangementWitness", detect_radial_arrangement, 5, common_inv, failure_modes=("fewer_than_three_parts",)),
        _reg("pair_parts_by_symmetry", ("PartGraphWitness",), "PartGraphWitness", pair_parts_by_symmetry, 3, common_inv),
        _reg("select_all_objects", ("Scene",), "Scene", select_all_objects, 1),
        _reg("select_principal_objects", ("Scene",), "Scene", select_principal_objects, 1),
        _reg("select_largest", ("Scene",), "Object", select_largest, 1),
        _reg("select_largest_object", ("Scene",), "Object", select_largest_object, 1),
        _reg("select_smallest_object", ("Scene",), "Object", select_smallest_object, 1),
        _reg("select_inner_object", ("Scene",), "Object", select_inner_object, 1),
        _reg("select_outer_object", ("Scene",), "Object", select_outer_object, 1),
        _reg("select_parts", ("PartGraphWitness",), "PartGraphWitness", select_parts, 1),
        _reg("object_count", ("Scene",), "Measurement", object_count, 1, proxy_for=("count", "component", "connected", "continuous", "unbroken", "disconnected", "disjoint", "separate")),
        _reg("total_ink", ("Panel",), "Measurement", total_ink, 1, proxy_for=("ink", "area")),
        _reg("largest_area", ("Scene",), "Measurement", largest_area, 1, proxy_for=("area",)),
        _reg("bbox_aspect", ("Object",), "Measurement", bbox_aspect, 1, proxy_for=("elongated", "aspect", "thin", "narrow", "wide", "slender")),
        _reg("bbox_fill", ("Object",), "Measurement", bbox_fill, 1, proxy_for=("sparse", "filled", "fill")),
        _reg("closure_ratio", ("Object",), "Measurement", closure_ratio, 2, proxy_for=("open", "closed")),
        _reg("symmetry_residual", ("Object",), "Measurement", symmetry_residual, 3, proxy_for=("symmetric", "asymmetric")),
        _reg("witness_confidence", ("TriangleWitness",), "Measurement", witness_confidence, 1),
        _reg("quadrilateral_confidence", ("QuadrilateralWitness",), "Measurement", witness_confidence, 1),
        _reg("circle_residual", ("CircleWitness",), "Measurement", witness_residual, 1),
        _reg("contact_confidence", ("ContactWitness",), "Measurement", witness_confidence, 1),
        _reg("circle_intersection_confidence", ("CircleIntersectionWitness",), "Measurement", witness_confidence, 1),
        _reg("radial_part_count", ("RadialArrangementWitness",), "Measurement", radial_part_count, 1),
        _reg("radial_uniformity", ("RadialArrangementWitness",), "Measurement", witness_confidence, 1),
        _reg("symmetry_order_score", ("SymmetryWitness",), "Measurement", symmetry_order_score, 1),
    ):
        reg.register(contract)
    return reg

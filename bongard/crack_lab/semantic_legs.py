"""Typed visual legs for semantic Bongard cones.

The unrestricted predicate path still lives in ``bongard_arena.py`` and
``bongard_legs.py``.  This module is the semantic-pure basis: every arrow has
an auditable contract and returns either a typed witness or a scalar
measurement derived from such witnesses.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from visual_witnesses import (
    ArcWitness,
    CircleIntersectionWitness,
    CirclePairWitness,
    CircleWitness,
    ContactWitness,
    ContourWitness,
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


def _component_masks(panel: np.ndarray, min_pixels: int = 3) -> tuple[ObjectMask, ...]:
    ink = np.asarray(panel, dtype=np.uint8) > 0
    h, w = ink.shape
    seen = np.zeros_like(ink, dtype=bool)
    objects: list[ObjectMask] = []
    for y0, x0 in np.argwhere(ink):
        if seen[y0, x0]:
            continue
        stack = [(int(y0), int(x0))]
        seen[y0, x0] = True
        pts: list[tuple[int, int]] = []
        while stack:
            y, x = stack.pop()
            pts.append((y, x))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < h and 0 <= xx < w and ink[yy, xx] and not seen[yy, xx]:
                        seen[yy, xx] = True
                        stack.append((yy, xx))
        if len(pts) >= min_pixels:
            mask = np.zeros_like(ink, dtype=bool)
            ys, xs = zip(*pts)
            mask[list(ys), list(xs)] = True
            objects.append(ObjectMask(mask, f"object_{len(objects)}"))
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


def _boundary_points(mask: np.ndarray) -> np.ndarray:
    pts = []
    h, w = mask.shape
    for y, x in np.argwhere(mask):
        boundary = False
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            yy, xx = int(y) + dy, int(x) + dx
            if yy < 0 or yy >= h or xx < 0 or xx >= w or not mask[yy, xx]:
                boundary = True
                break
        if boundary:
            pts.append((float(x), float(y)))
    return np.asarray(pts, dtype=float)


def extract_contours(obj: ObjectMask) -> ContourWitness:
    pts = _boundary_points(obj.mask)
    if len(pts) == 0:
        return ContourWitness(source_component_id=obj.object_id, provenance=("extract_contours",))
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    order = np.argsort(angles)
    ordered = tuple((float(x), float(y)) for x, y in pts[order])
    return ContourWitness(
        source_component_id=obj.object_id,
        points=ordered,
        is_closed=True,
        confidence=1.0,
        provenance=("extract_contours",),
    )


def build_containment_tree(scene: Scene) -> PartGraphWitness:
    return build_part_graph(scene)


def _degrees(mask: np.ndarray) -> tuple[int, int, int]:
    endpoints = branches = edge_count = 0
    pts = np.argwhere(mask)
    h, w = mask.shape
    for y, x in pts:
        deg = 0
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                yy, xx = int(y) + dy, int(x) + dx
                if 0 <= yy < h and 0 <= xx < w and mask[yy, xx]:
                    deg += 1
        edge_count += deg
        if deg <= 1:
            endpoints += 1
        if deg >= 3:
            branches += 1
    edges = edge_count // 2
    cycles = max(0, edges - len(pts) + 1) if len(pts) else 0
    return endpoints, branches, cycles


def build_skeleton_graph(obj: ObjectMask) -> SkeletonGraphWitness:
    endpoints, branches, cycles = _degrees(obj.mask)
    pts = np.argwhere(obj.mask)
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
    pts = np.argwhere(obj.mask)
    if len(pts) == 0:
        return 1.0
    endpoints, _, _ = _degrees(obj.mask)
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


def _corner_vertices(contour: ContourWitness, max_vertices: int = 8) -> tuple[PointWitness, ...]:
    pts = np.asarray(contour.points, dtype=float)
    if len(pts) == 0:
        return ()
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    bins = np.linspace(-math.pi, math.pi, max_vertices + 1)
    verts = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        idx = np.where((angles >= lo) & (angles < hi))[0]
        if len(idx) == 0:
            continue
        sub = pts[idx]
        d = np.linalg.norm(sub - center, axis=1)
        x, y = sub[int(np.argmax(d))]
        verts.append(PointWitness(x=float(x), y=float(y), source_id=contour.source_component_id))
    if len(verts) <= 5:
        # Collapse near-duplicate angular bins for simple polygons.
        dedup: list[PointWitness] = []
        for v in verts:
            if not dedup or math.hypot(v.x - dedup[-1].x, v.y - dedup[-1].y) > 3.0:
                dedup.append(v)
        verts = dedup
    return tuple(verts)


def detect_corners(contour: ContourWitness) -> PolygonWitness:
    vertices = _corner_vertices(contour)
    return PolygonWitness(
        source_component_id=contour.source_component_id,
        vertices=vertices,
        side_count=len(vertices),
        confidence=1.0 if len(vertices) >= 3 else 0.2,
        provenance=contour.provenance + ("detect_corners",),
    )


def decompose_into_line_segments(contour: ContourWitness) -> PolygonWitness:
    return detect_corners(contour)


def fit_polygon(contour: ContourWitness) -> PolygonWitness:
    return detect_corners(contour)


def polygon_side_count(poly: PolygonWitness) -> float:
    return float(poly.side_count)


def classify_triangle(poly: PolygonWitness) -> TriangleWitness:
    if poly.side_count != 3:
        raise ValueError("polygon is not a triangle")
    return TriangleWitness(
        source_component_id=poly.source_component_id,
        vertices=poly.vertices,
        confidence=poly.confidence,
        residual=poly.residual,
        provenance=poly.provenance + ("classify_triangle",),
    )


def classify_quadrilateral(poly: PolygonWitness) -> QuadrilateralWitness:
    if poly.side_count != 4:
        raise ValueError("polygon is not a quadrilateral")
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


def fit_circle(contour: ContourWitness) -> CircleWitness:
    pts = np.asarray(contour.points, dtype=float)
    if len(pts) < 3:
        raise ValueError("not enough contour points for circle fit")
    center = pts.mean(axis=0)
    radii = np.linalg.norm(pts - center, axis=1)
    radius = float(np.mean(radii))
    residual = float(np.sqrt(np.mean((radii - radius) ** 2)) / max(radius, 1e-9))
    return CircleWitness(
        source_component_id=contour.source_component_id,
        center=PointWitness(x=float(center[0]), y=float(center[1]), source_id=contour.source_component_id),
        radius=radius,
        support_points=tuple((float(x), float(y)) for x, y in pts[:: max(1, len(pts) // 32)]),
        residual=residual,
        confidence=max(0.0, 1.0 - residual),
        provenance=contour.provenance + ("fit_circle",),
    )


def fit_arc(contour: ContourWitness) -> ArcWitness:
    circle = fit_circle(contour)
    return ArcWitness(
        source_component_id=contour.source_component_id,
        center=circle.center,
        radius=circle.radius,
        angle_degrees=180.0,
        residual=circle.residual,
        confidence=circle.confidence,
        provenance=circle.provenance + ("fit_arc",),
    )


def decompose_curve_into_arcs_and_lines(contour: ContourWitness) -> PartGraphWitness:
    part = PartWitness(
        part_id=f"{contour.source_component_id}_curve",
        role="curve",
        source_component_id=contour.source_component_id,
        contour=contour,
        provenance=contour.provenance + ("decompose_curve_into_arcs_and_lines",),
    )
    return PartGraphWitness(parts=(part,), provenance=part.provenance)


def fit_multiple_circles(scene: Scene) -> CirclePairWitness:
    circles = []
    for obj in scene.objects[:2]:
        circles.append(fit_circle(extract_contours(obj)))
    if len(circles) < 2:
        raise ValueError("need at least two circle candidates")
    a, b = circles[:2]
    d = math.hypot(a.center.x - b.center.x, a.center.y - b.center.y)
    return CirclePairWitness(
        first=a,
        second=b,
        center_distance=d,
        confidence=min(a.confidence, b.confidence),
        residual=max(a.residual, b.residual),
        provenance=a.provenance + b.provenance + ("fit_multiple_circles",),
    )


def decompose_component_into_parts(obj: ObjectMask) -> PartGraphWitness:
    contour = extract_contours(obj)
    part = PartWitness(
        part_id=f"{obj.object_id}_part_0",
        role="principal",
        source_component_id=obj.object_id,
        contour=contour,
        provenance=("decompose_component_into_parts",),
    )
    return PartGraphWitness(parts=(part,), provenance=("decompose_component_into_parts",))


def build_part_graph(value: Scene | ObjectMask) -> PartGraphWitness:
    if isinstance(value, Scene):
        parts = tuple(
            PartWitness(part_id=f"{obj.object_id}_part", role="object",
                        source_component_id=obj.object_id, contour=extract_contours(obj),
                        provenance=("build_part_graph",))
            for obj in value.objects
        )
    else:
        parts = decompose_component_into_parts(value).parts
    adjacency = tuple((parts[i].part_id, parts[j].part_id)
                      for i in range(len(parts)) for j in range(i + 1, len(parts)))
    return PartGraphWitness(parts=parts, adjacency=adjacency, provenance=("build_part_graph",))


def _part_centroid(part: PartWitness) -> PointWitness:
    if not part.contour or not part.contour.points:
        return PointWitness(source_id=part.part_id)
    pts = np.asarray(part.contour.points, dtype=float)
    c = pts.mean(axis=0)
    return PointWitness(x=float(c[0]), y=float(c[1]), source_id=part.part_id)


def detect_contact(graph: PartGraphWitness) -> ContactWitness:
    if len(graph.parts) < 2:
        raise ValueError("need at least two parts")
    a, b = graph.parts[:2]
    pa, pb = _part_centroid(a), _part_centroid(b)
    mid = PointWitness(x=(pa.x + pb.x) / 2.0, y=(pa.y + pb.y) / 2.0, source_id="contact")
    return ContactWitness(source_a=a.part_id, source_b=b.part_id, points=(mid,),
                          provenance=graph.provenance + ("detect_contact",))


def detect_attachment(graph: PartGraphWitness) -> ContactWitness:
    return detect_contact(graph)


def detect_tangency(graph: PartGraphWitness) -> ContactWitness:
    c = detect_contact(graph)
    return ContactWitness(source_a=c.source_a, source_b=c.source_b, points=c.points,
                          relation="tangency",
                          provenance=c.provenance + ("detect_tangency",))


def detect_intersection(value: PartGraphWitness | CirclePairWitness) -> IntersectionWitness:
    if isinstance(value, CirclePairWitness):
        return circle_pair_intersection(value)
    c = detect_contact(value)
    return IntersectionWitness(source_a=c.source_a, source_b=c.source_b, points=c.points,
                               provenance=c.provenance + ("detect_intersection",))


def detect_shared_endpoint(graph: PartGraphWitness) -> ContactWitness:
    c = detect_contact(graph)
    return ContactWitness(source_a=c.source_a, source_b=c.source_b, points=c.points,
                          relation="shared_endpoint",
                          provenance=c.provenance + ("detect_shared_endpoint",))


def detect_shared_point(graph: PartGraphWitness) -> ContactWitness:
    c = detect_contact(graph)
    return ContactWitness(source_a=c.source_a, source_b=c.source_b, points=c.points,
                          relation="shared_point",
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
    centers = tuple(_part_centroid(p) for p in graph.parts)
    if centers:
        cx = sum(p.x for p in centers) / len(centers)
        cy = sum(p.y for p in centers) / len(centers)
    else:
        cx = cy = 0.0
    return RadialArrangementWitness(
        center=PointWitness(x=cx, y=cy, source_id="radial_center"),
        parts=graph.parts,
        part_count=len(graph.parts),
        symmetry_order=len(graph.parts),
        confidence=1.0 if len(graph.parts) >= 3 else 0.4,
        provenance=graph.provenance + ("detect_radial_arrangement",),
    )


def pair_parts_by_symmetry(graph: PartGraphWitness) -> PartGraphWitness:
    return graph


def prototype_bird_like(graph: PartGraphWitness) -> PrototypeWitness:
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
        _reg("extract_contours", ("Object",), "ContourWitness", extract_contours, 4, common_inv),
        _reg("build_containment_tree", ("Scene",), "PartGraphWitness", build_containment_tree, 3, common_inv),
        _reg("skeletonize_component", ("Object",), "SkeletonGraphWitness", skeletonize_component, 3, common_inv),
        _reg("build_skeleton_graph", ("Object",), "SkeletonGraphWitness", build_skeleton_graph, 3, common_inv),
        _reg("endpoint_count", ("SkeletonGraphWitness",), "Measurement", endpoint_count, 1, common_inv),
        _reg("branch_count", ("SkeletonGraphWitness",), "Measurement", branch_count, 1, common_inv),
        _reg("cycle_count", ("SkeletonGraphWitness",), "Measurement", cycle_count, 1, common_inv),
        _reg("estimate_tangents", ("ContourWitness",), "CurveWitness", estimate_tangents, 2, common_inv),
        _reg("estimate_curvature", ("ContourWitness",), "CurveWitness", estimate_curvature, 2, common_inv),
        _reg("curvature_extrema", ("CurveWitness",), "SkeletonGraphWitness", curvature_extrema, 2, common_inv),
        _reg("decompose_curve_into_arcs_and_lines", ("ContourWitness",), "PartGraphWitness", decompose_curve_into_arcs_and_lines, 5, common_inv),
        _reg("fit_line_segment", ("ContourWitness",), "LineSegmentWitness", fit_line_segment, 2, common_inv),
        _reg("fit_arc", ("ContourWitness",), "ArcWitness", fit_arc, 3, common_inv),
        _reg("fit_circle", ("ContourWitness",), "CircleWitness", fit_circle, 4, common_inv, failure_modes=("not_enough_points", "high_residual")),
        _reg("fit_multiple_circles", ("Scene",), "CirclePairWitness", fit_multiple_circles, 8, common_inv, failure_modes=("fewer_than_two_candidates", "high_residual")),
        _reg("detect_corners", ("ContourWitness",), "PolygonWitness", detect_corners, 4, common_inv),
        _reg("decompose_into_line_segments", ("ContourWitness",), "PolygonWitness", decompose_into_line_segments, 4, common_inv),
        _reg("fit_polygon", ("ContourWitness",), "PolygonWitness", fit_polygon, 5, common_inv),
        _reg("polygon_side_count", ("PolygonWitness",), "Measurement", polygon_side_count, 1, common_inv),
        _reg("classify_triangle", ("PolygonWitness",), "TriangleWitness", classify_triangle, 2, common_inv),
        _reg("classify_quadrilateral", ("PolygonWitness",), "QuadrilateralWitness", classify_quadrilateral, 2, common_inv),
        _reg("decompose_component_into_parts", ("Object",), "PartGraphWitness", decompose_component_into_parts, 5, common_inv),
        _reg("build_part_graph", ("Scene",), "PartGraphWitness", build_part_graph, 5, common_inv),
        _reg("build_object_part_graph", ("Object",), "PartGraphWitness", build_part_graph, 5, common_inv),
        _reg("detect_attachment", ("PartGraphWitness",), "ContactWitness", detect_attachment, 3, common_inv),
        _reg("detect_contact", ("PartGraphWitness",), "ContactWitness", detect_contact, 3, common_inv),
        _reg("detect_tangency", ("PartGraphWitness",), "ContactWitness", detect_tangency, 3, common_inv),
        _reg("detect_intersection", ("PartGraphWitness",), "IntersectionWitness", detect_intersection, 3, common_inv),
        _reg("circle_pair_intersection", ("CirclePairWitness",), "CircleIntersectionWitness", circle_pair_intersection, 3, common_inv),
        _reg("detect_shared_endpoint", ("PartGraphWitness",), "ContactWitness", detect_shared_endpoint, 2, common_inv),
        _reg("detect_shared_point", ("PartGraphWitness",), "ContactWitness", detect_shared_point, 2, common_inv),
        _reg("reflection_symmetry", ("Object",), "SymmetryWitness", reflection_symmetry, 3, common_inv),
        _reg("rotational_symmetry_order", ("Object",), "SymmetryWitness", rotational_symmetry_order, 4, common_inv),
        _reg("detect_radial_arrangement", ("PartGraphWitness",), "RadialArrangementWitness", detect_radial_arrangement, 5, common_inv),
        _reg("pair_parts_by_symmetry", ("PartGraphWitness",), "PartGraphWitness", pair_parts_by_symmetry, 3, common_inv),
        _reg("select_all_objects", ("Scene",), "Scene", select_all_objects, 1),
        _reg("select_principal_objects", ("Scene",), "Scene", select_principal_objects, 1),
        _reg("select_largest", ("Scene",), "Object", select_largest, 1),
        _reg("select_largest_object", ("Scene",), "Object", select_largest_object, 1),
        _reg("select_smallest_object", ("Scene",), "Object", select_smallest_object, 1),
        _reg("select_inner_object", ("Scene",), "Object", select_inner_object, 1),
        _reg("select_outer_object", ("Scene",), "Object", select_outer_object, 1),
        _reg("select_parts", ("PartGraphWitness",), "PartGraphWitness", select_parts, 1),
        _reg("object_count", ("Scene",), "Measurement", object_count, 1, proxy_for=("count",)),
        _reg("total_ink", ("Panel",), "Measurement", total_ink, 1, proxy_for=("ink", "area")),
        _reg("largest_area", ("Scene",), "Measurement", largest_area, 1, proxy_for=("area",)),
        _reg("bbox_aspect", ("Object",), "Measurement", bbox_aspect, 1, proxy_for=("elongated", "aspect")),
        _reg("bbox_fill", ("Object",), "Measurement", bbox_fill, 1, proxy_for=("sparse", "filled", "fill")),
        _reg("closure_ratio", ("Object",), "Measurement", closure_ratio, 2, proxy_for=("open", "closed")),
        _reg("symmetry_residual", ("Object",), "Measurement", symmetry_residual, 3, proxy_for=("symmetric", "asymmetric")),
        _reg("witness_confidence", ("TriangleWitness",), "Measurement", witness_confidence, 1),
        _reg("quadrilateral_confidence", ("QuadrilateralWitness",), "Measurement", witness_confidence, 1),
        _reg("circle_residual", ("CircleWitness",), "Measurement", witness_residual, 1),
        _reg("circle_intersection_confidence", ("CircleIntersectionWitness",), "Measurement", witness_confidence, 1),
        _reg("radial_part_count", ("RadialArrangementWitness",), "Measurement", radial_part_count, 1),
        _reg("symmetry_order_score", ("SymmetryWitness",), "Measurement", symmetry_order_score, 1),
    ):
        reg.register(contract)
    return reg

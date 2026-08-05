"""Registry-wide contract laboratory for proposer-visible semantic legs.

This module intentionally tests the public ``LegContract`` surface rather
than compiler behavior.  Every registered implementation receives a direct
constructive input, every advertised failure mode has an adversarial probe,
and every claimed invariance is exercised on the leg that claims it.
"""
from __future__ import annotations

import dataclasses
import math
import os
import re
import sys
from numbers import Real
from typing import Any

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import semantic_legs as L
from visual_witnesses import (
    AngleWitness,
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
    PointContactSignature,
    PointWitness,
    PolygonWitness,
    QuadrilateralWitness,
    RadialArrangementWitness,
    SkeletonGraphWitness,
    TriangleWitness,
    Witness,
)


REGISTRY = L.default_registry()
CONTRACTS = {contract.name: contract for contract in REGISTRY.contracts()}


def _panel_with_two_blobs() -> np.ndarray:
    panel = np.zeros((96, 96), dtype=np.uint8)
    panel[18:31, 20:35] = 1
    panel[58:78, 61:82] = 1
    return panel


def _point_contact_panel() -> np.ndarray:
    """Two square loops with one diagonal point contact."""
    panel = np.zeros((96, 96), dtype=np.uint8)
    for low, high in ((15, 45), (46, 76)):
        panel[low, low:high + 1] = 1
        panel[high, low:high + 1] = 1
        panel[low:high + 1, low] = 1
        panel[low:high + 1, high] = 1
    return panel


def _stroke_object(kind: str = "square") -> L.ObjectMask:
    mask = np.zeros((96, 96), dtype=bool)
    if kind == "cross":
        mask[15:81, 47] = True
        mask[48, 15:81] = True
    elif kind == "line":
        mask[40, 18:78] = True
    else:
        mask[20, 20:76] = True
        mask[75, 20:76] = True
        mask[20:76, 20] = True
        mask[20:76, 75] = True
    return L.ObjectMask(mask, kind)


def _circle_contour(*, closed: bool = True) -> ContourWitness:
    stop = 2.0 * math.pi if closed else 1.35 * math.pi
    theta = np.linspace(0.0, stop, 120, endpoint=not closed)
    points = tuple((31.0 + 14.0 * math.cos(t),
                    27.0 + 14.0 * math.sin(t)) for t in theta)
    return ContourWitness(
        source_component_id="circle",
        points=points,
        is_closed=closed,
        confidence=0.93,
        provenance=("fixture",),
    )


def _line_contour() -> ContourWitness:
    return ContourWitness(
        source_component_id="line",
        points=tuple((8.0 + i, 17.0 + 0.25 * i) for i in range(48)),
        is_closed=False,
        confidence=0.95,
        provenance=("fixture",),
    )


def _square_contour() -> ContourWitness:
    points: list[tuple[float, float]] = []
    corners = ((12.0, 12.0), (52.0, 12.0), (52.0, 52.0),
               (12.0, 52.0), (12.0, 12.0))
    for (x0, y0), (x1, y1) in zip(corners[:-1], corners[1:]):
        points.extend(
            (x0 + (x1 - x0) * t, y0 + (y1 - y0) * t)
            for t in np.linspace(0.0, 1.0, 25, endpoint=False)
        )
    return ContourWitness(
        source_component_id="square",
        points=tuple(points),
        is_closed=True,
        confidence=0.91,
        residual=0.02,
        provenance=("fixture",),
    )


def _wavy_closed_contour() -> ContourWitness:
    theta = np.linspace(0.0, 2.0 * math.pi, 128, endpoint=False)
    radius = 30.0 + 10.0 * np.sin(7.0 * theta)
    return ContourWitness(
        source_component_id="wavy",
        points=tuple(zip(64.0 + radius * np.cos(theta),
                         64.0 + radius * np.sin(theta))),
        is_closed=True,
        provenance=("fixture",),
    )


def _polygon(side_count: int) -> PolygonWitness:
    vertices = tuple(
        PointWitness(
            x=30.0 + 15.0 * math.cos(2.0 * math.pi * i / side_count),
            y=30.0 + 15.0 * math.sin(2.0 * math.pi * i / side_count),
            source_id="polygon",
        )
        for i in range(side_count)
    )
    return PolygonWitness(
        source_component_id="polygon",
        vertices=vertices,
        side_count=side_count,
        confidence=0.87,
        residual=0.03,
        provenance=("fixture",),
    )


def _part(part_id: str, x: float, y: float, size: float = 6.0) -> PartWitness:
    contour = ContourWitness(
        source_component_id="fixture",
        points=((x - size, y), (x, y + size), (x + size, y)),
        is_closed=False,
        provenance=("fixture",),
    )
    return PartWitness(
        part_id=part_id,
        role="stroke",
        source_component_id="fixture",
        contour=contour,
        provenance=("fixture",),
    )


def _contact_graph(*, intersection: bool = False) -> PartGraphWitness:
    parts = (_part("left", 18.0, 30.0), _part("right", 42.0, 30.0))
    cls = IntersectionWitness if intersection else ContactWitness
    relation = "intersection" if intersection else "attachment"
    contact = cls(
        source_a="left",
        source_b="right",
        points=(PointWitness(x=30.0, y=30.0, source_id="fixture"),),
        relation=relation,
        confidence=0.84,
        provenance=("fixture",),
    )
    return PartGraphWitness(
        parts=parts,
        contacts=(contact,),
        adjacency=(("left", "right"),),
        confidence=0.9,
        provenance=("fixture",),
    )


def _junction_angle_part(
        part_id: str, first_angle: float, second_angle: float,
        ) -> PartWitness:
    def polar(angle_degrees: float, radius: float) -> tuple[float, float]:
        angle = math.radians(angle_degrees)
        return radius * math.cos(angle), radius * math.sin(angle)

    points = [
        polar(first_angle, radius)
        for radius in np.linspace(1.0, 30.0, 30)
    ]
    points.extend(
        polar(angle, 30.0)
        for angle in np.linspace(first_angle, second_angle, 61)[1:-1]
    )
    points.extend(
        polar(second_angle, radius)
        for radius in np.linspace(30.0, 1.0, 30)
    )
    contour = ContourWitness(
        source_component_id=part_id,
        points=tuple(points),
        is_closed=False,
        confidence=0.98,
        provenance=("junction-angle-fixture",),
    )
    return PartWitness(
        part_id=part_id,
        role="stroke-loop",
        source_component_id=part_id,
        contour=contour,
        confidence=0.98,
        provenance=("junction-angle-fixture",),
    )


def _junction_angle_graph() -> PartGraphWitness:
    parts = (
        _junction_angle_part("angle-a", 0.0, 140.0),
        _junction_angle_part("angle-b", 30.0, 250.0),
    )
    junction = PointWitness(x=0.0, y=0.0, source_id="junction")
    return PartGraphWitness(
        parts=parts,
        contacts=(IntersectionWitness(
            source_a=parts[0].part_id,
            source_b=parts[1].part_id,
            points=(junction,),
            relation="intersection",
            confidence=0.97,
            provenance=("junction-angle-fixture",),
        ),),
        adjacency=((parts[0].part_id, parts[1].part_id),),
        confidence=0.96,
        provenance=("junction-angle-fixture",),
    )


def _high_residual_junction_graph() -> PartGraphWitness:
    graph = _junction_angle_graph()
    first = graph.parts[0]
    assert first.contour is not None
    zigzag = tuple(
        (float(index), 5.0 if index % 2 else -5.0)
        for index in range(1, 19)
    )
    contour = dataclasses.replace(
        first.contour,
        points=((0.5, 0.0),) + zigzag + first.contour.points[18:],
    )
    return dataclasses.replace(
        graph,
        parts=(dataclasses.replace(first, contour=contour), graph.parts[1]),
    )


def _radial_graph() -> PartGraphWitness:
    parts = tuple(
        _part(f"ray_{i}", 40.0 + 18.0 * math.cos(2.0 * math.pi * i / 3),
              40.0 + 18.0 * math.sin(2.0 * math.pi * i / 3), 3.0)
        for i in range(3)
    )
    hub = PointWitness(x=40.0, y=40.0, source_id="hub")
    return PartGraphWitness(
        parts=parts,
        contacts=(ContactWitness(
            source_a="ray_0", source_b="ray_1", points=(hub,),
            relation="attachment", provenance=("fixture",),
        ),),
        adjacency=(("ray_0", "ray_1"), ("ray_0", "ray_2"),
                   ("ray_1", "ray_2")),
        provenance=("fixture",),
    )


def _circle(source: str, x: float, y: float, radius: float) -> CircleWitness:
    return CircleWitness(
        source_component_id=source,
        center=PointWitness(x=x, y=y, source_id=source),
        radius=radius,
        residual=0.015,
        confidence=0.94,
        provenance=("fixture",),
    )


def _circle_pair() -> CirclePairWitness:
    return CirclePairWitness(
        first=_circle("first", 0.0, 0.0, 5.0),
        second=_circle("second", 6.0, 0.0, 5.0),
        center_distance=6.0,
        confidence=0.92,
        residual=0.02,
        provenance=("fixture",),
    )


def _two_circle_scene() -> L.Scene:
    shape = (128, 128)
    yy, xx = np.indices(shape)
    masks = []
    for idx, (cx, cy, radius) in enumerate(((35, 52, 15), (89, 72, 19))):
        distance = np.hypot(xx - cx, yy - cy)
        mask = np.abs(distance - radius) <= 0.7
        masks.append(L.ObjectMask(mask, f"circle_{idx}"))
    panel = np.logical_or.reduce([obj.mask for obj in masks]).astype(np.uint8)
    return L.Scene(panel, tuple(masks))


def _skeleton_witness() -> SkeletonGraphWitness:
    return SkeletonGraphWitness(
        source_component_id="skeleton",
        nodes=(PointWitness(x=10.0, y=11.0, source_id="skeleton"),),
        endpoint_count=4,
        branch_count=1,
        cycle_count=0,
        confidence=0.89,
        provenance=("fixture",),
    )


def _arc_witness() -> ArcWitness:
    return ArcWitness(
        source_component_id="arc",
        points=((4.0, 5.0), (8.0, 9.0), (12.0, 5.0)),
        center=PointWitness(x=8.0, y=5.0, source_id="arc"),
        radius=4.0,
        angle_degrees=180.0,
        confidence=0.88,
        residual=0.025,
        provenance=("fixture",),
    )


def _circle_intersection() -> CircleIntersectionWitness:
    return L.circle_pair_intersection(_circle_pair())


def _fixture_for(contract_name: str) -> Any:
    domain = CONTRACTS[contract_name].domain[0]
    if contract_name == "extract_point_contact_signature":
        return _point_contact_panel()
    if contract_name == "minimum_incident_angle":
        return _junction_angle_graph()
    if contract_name == "fit_multiple_circles":
        return _two_circle_scene()
    if contract_name in {"fit_circle"}:
        return _circle_contour(closed=True)
    if contract_name in {"fit_arc"}:
        return _circle_contour(closed=False)
    if contract_name == "fit_line_segment":
        return _line_contour()
    if contract_name in {
        "fit_polygon", "detect_corners", "decompose_into_line_segments",
        "decompose_curve_into_arcs_and_lines",
    }:
        return _square_contour()
    if contract_name == "classify_quadrilateral":
        return _polygon(4)
    if contract_name in {"detect_intersection", "intersection_count"}:
        return _contact_graph(intersection=True)
    if contract_name == "detect_radial_arrangement":
        return _radial_graph()
    if domain == "Panel":
        return _panel_with_two_blobs()
    if domain == "Scene":
        return L.parse_scene(_panel_with_two_blobs())
    if domain == "Object":
        kind = "cross" if "part_graph" in contract_name or "parts" in contract_name else "square"
        return _stroke_object(kind)
    if domain == "ContourWitness":
        return _square_contour()
    if domain == "SkeletonGraphWitness":
        return _skeleton_witness()
    if domain == "ArcWitness":
        return _arc_witness()
    if domain == "AngleWitness":
        return L.minimum_incident_angle(_junction_angle_graph())
    if domain == "PointContactSignature":
        return L.extract_point_contact_signature(_point_contact_panel())
    if domain == "LineSegmentWitness":
        return L.fit_line_segment(_line_contour())
    if domain == "PolygonWitness":
        return _polygon(3)
    if domain == "PartGraphWitness":
        return _contact_graph()
    if domain == "CirclePairWitness":
        return _circle_pair()
    if domain == "TriangleWitness":
        poly = _polygon(3)
        return TriangleWitness(
            source_component_id=poly.source_component_id,
            vertices=poly.vertices,
            confidence=poly.confidence,
            residual=poly.residual,
            provenance=poly.provenance,
        )
    if domain == "QuadrilateralWitness":
        poly = _polygon(4)
        return QuadrilateralWitness(
            source_component_id=poly.source_component_id,
            vertices=poly.vertices,
            confidence=poly.confidence,
            residual=poly.residual,
            provenance=poly.provenance,
        )
    if domain == "CircleWitness":
        return _circle("circle", 20.0, 30.0, 7.0)
    if domain == "ContactWitness":
        return _contact_graph().contacts[0]
    if domain == "IntersectionWitness":
        return L.detect_intersection(_contact_graph(intersection=True))
    if domain == "CircleIntersectionWitness":
        return _circle_intersection()
    if domain == "RadialArrangementWitness":
        return L.detect_radial_arrangement(_radial_graph())
    raise AssertionError(f"no constructive fixture for {contract_name}: {domain}")


CODOMAIN_TYPES = {
    "BinaryPanel": np.ndarray,
    "Scene": L.Scene,
    "Object": L.ObjectMask,
    "ContourWitness": ContourWitness,
    "SkeletonGraphWitness": SkeletonGraphWitness,
    "CurveWitness": CurveWitness,
    "PartGraphWitness": PartGraphWitness,
    "PartWitness": PartWitness,
    "LineSegmentWitness": LineSegmentWitness,
    "AngleWitness": AngleWitness,
    "PointContactSignature": PointContactSignature,
    "ArcWitness": ArcWitness,
    "CircleWitness": CircleWitness,
    "CirclePairWitness": CirclePairWitness,
    "CircleIntersectionWitness": CircleIntersectionWitness,
    "PolygonWitness": PolygonWitness,
    "TriangleWitness": TriangleWitness,
    "QuadrilateralWitness": QuadrilateralWitness,
    "ContactWitness": ContactWitness,
    "IntersectionWitness": IntersectionWitness,
    "RadialArrangementWitness": RadialArrangementWitness,
}


def test_registry_metadata_is_complete_normalized_and_typed():
    contracts = REGISTRY.contracts()
    names = [contract.name for contract in contracts]
    known_types = {
        "Panel", "BinaryPanel", "Scene", "Object", "Measurement",
        *CODOMAIN_TYPES,
    }
    assert names == sorted(names)
    assert len(names) == len(set(names))
    assert set(REGISTRY.names()) == set(names)
    assert set(REGISTRY.terminal_types()) == {c.codomain for c in contracts}

    for contract in contracts:
        assert re.fullmatch(r"[a-z][a-z0-9_]*", contract.name)
        assert contract.domain and all(item in known_types for item in contract.domain)
        assert contract.codomain in known_types
        assert callable(contract.implementation)
        assert type(contract.complexity) is int and contract.complexity > 0
        assert re.fullmatch(r"\d+\.\d+", contract.version)
        assert contract.invariances <= {
            "translation", "uniform_scale", "rotation", "reflection",
        }
        assert contract.equivariances <= {
            "translation", "uniform_scale", "rotation", "reflection",
        }
        assert contract.invariances.isdisjoint(contract.equivariances)
        assert len(contract.failure_modes) == len(set(contract.failure_modes))
        assert all(re.fullmatch(r"[a-z][a-z0-9_]*", mode)
                   for mode in contract.failure_modes)
        assert len(contract.indeterminate_modes) == len(
            set(contract.indeterminate_modes))
        assert all(re.fullmatch(r"[a-z][a-z0-9_]*", mode)
                   for mode in contract.indeterminate_modes)
        assert set(contract.failure_modes).isdisjoint(
            contract.indeterminate_modes)
        assert len(contract.proxy_for) == len(set(contract.proxy_for))
        assert all(term == term.strip().lower() and term for term in contract.proxy_for)
        if contract.codomain == "Measurement":
            assert contract.measurement_kind in {"continuous", "count", "binary"}
        else:
            assert contract.measurement_kind is None
        assert all(term in contract.proxy_for and direction in {"low", "high"}
                   for term, direction in contract.proxy_directions)

    assert set(CONTRACTS["part_count"].proxy_for) == {"part", "parts"}
    assert "adjacent" not in CONTRACTS["detect_contact"].proxy_for
    assert "overlap" not in CONTRACTS["detect_intersection"].proxy_for


@pytest.mark.parametrize("contract", REGISTRY.contracts(), ids=lambda c: c.name)
def test_every_registered_leg_executes_and_returns_its_codomain(contract):
    result = contract.implementation(_fixture_for(contract.name))
    if contract.codomain == "Measurement":
        assert isinstance(result, Real) and not isinstance(result, (bool, np.bool_))
        assert math.isfinite(float(result))
    elif contract.codomain == "BinaryPanel":
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.uint8
        assert set(np.unique(result)) <= {0, 1}
    else:
        assert isinstance(result, CODOMAIN_TYPES[contract.codomain])


def _shift_or_scale_mask(mask: np.ndarray, action: str) -> np.ndarray:
    if action == "translation":
        result = np.zeros_like(mask)
        result[7:, 11:] = mask[:-7, :-11]
        return result
    if action == "uniform_scale":
        return np.repeat(np.repeat(mask, 2, axis=0), 2, axis=1)
    if action == "rotation":
        return np.rot90(mask)
    if action == "reflection":
        return np.fliplr(mask)
    raise AssertionError(action)


def _xy(x: float, y: float, action: str) -> tuple[float, float]:
    if action == "translation":
        return x + 11.0, y + 7.0
    if action == "uniform_scale":
        return 2.0 * x, 2.0 * y
    if action == "rotation":
        return -y, x
    if action == "reflection":
        return -x, y
    raise AssertionError(action)


def _transform(value: Any, action: str) -> Any:
    if isinstance(value, np.ndarray):
        return _shift_or_scale_mask(value, action)
    if isinstance(value, L.ObjectMask):
        return L.ObjectMask(_shift_or_scale_mask(value.mask, action), value.object_id)
    if isinstance(value, L.Scene):
        return L.Scene(
            _shift_or_scale_mask(value.panel, action),
            tuple(_transform(obj, action) for obj in value.objects),
        )
    if isinstance(value, PointWitness):
        x, y = _xy(value.x, value.y, action)
        return dataclasses.replace(value, x=x, y=y)
    if isinstance(value, ContourWitness):
        return dataclasses.replace(
            value,
            points=tuple(_xy(x, y, action) for x, y in value.points),
        )
    if isinstance(value, SkeletonGraphWitness):
        return dataclasses.replace(
            value, nodes=tuple(_transform(p, action) for p in value.nodes))
    if isinstance(value, PolygonWitness):
        return dataclasses.replace(
            value, vertices=tuple(_transform(p, action) for p in value.vertices))
    if isinstance(value, PartWitness):
        contour = _transform(value.contour, action) if value.contour else None
        return dataclasses.replace(value, contour=contour)
    if isinstance(value, PartGraphWitness):
        return dataclasses.replace(
            value,
            parts=tuple(_transform(p, action) for p in value.parts),
            contacts=tuple(_transform(c, action) for c in value.contacts),
        )
    if isinstance(value, PointContactSignature):
        rays = []
        for ray in value.rays:
            segment = _transform(ray.segment, action)
            dx = segment.end.x - segment.start.x
            dy = segment.end.y - segment.start.y
            rays.append(dataclasses.replace(
                ray,
                segment=segment,
                direction_degrees=math.degrees(math.atan2(dy, dx)) % 360.0,
            ))
        return L._assemble_point_contact_signature(
            _transform(value.vertex, action),
            value.part_ids,
            tuple(rays),
            value.provenance,
            confidence=value.confidence,
        )
    if isinstance(value, RadialArrangementWitness):
        return dataclasses.replace(
            value,
            center=_transform(value.center, action),
            parts=tuple(_transform(part, action) for part in value.parts),
        )
    if isinstance(value, CirclePairWitness):
        factor = 2.0 if action == "uniform_scale" else 1.0
        return dataclasses.replace(
            value,
            first=_transform(value.first, action),
            second=_transform(value.second, action),
            center_distance=value.center_distance * factor,
        )
    if isinstance(value, CircleWitness):
        factor = 2.0 if action == "uniform_scale" else 1.0
        return dataclasses.replace(
            value,
            center=_transform(value.center, action),
            radius=value.radius * factor,
            support_points=tuple(_xy(x, y, action) for x, y in value.support_points),
        )
    if isinstance(value, CircleIntersectionWitness):
        return dataclasses.replace(
            value,
            points=tuple(_transform(point, action) for point in value.points),
            pair=_transform(value.pair, action),
        )
    if isinstance(value, ContactWitness):
        return dataclasses.replace(
            value, points=tuple(_transform(p, action) for p in value.points))
    if isinstance(value, AngleWitness):
        return dataclasses.replace(
            value, vertex=_transform(value.vertex, action))
    if isinstance(value, ArcWitness):
        factor = 2.0 if action == "uniform_scale" else 1.0
        return dataclasses.replace(
            value,
            points=tuple(_xy(x, y, action) for x, y in value.points),
            endpoints=tuple(_transform(p, action) for p in value.endpoints),
            center=_transform(value.center, action),
            radius=value.radius * factor,
        )
    if isinstance(value, LineSegmentWitness):
        factor = 2.0 if action == "uniform_scale" else 1.0
        return dataclasses.replace(
            value,
            points=tuple(_xy(x, y, action) for x, y in value.points),
            endpoints=tuple(_transform(p, action) for p in value.endpoints),
            start=_transform(value.start, action),
            end=_transform(value.end, action),
            length=value.length * factor,
        )
    if isinstance(value, CurveWitness):
        return dataclasses.replace(
            value,
            points=tuple(_xy(x, y, action) for x, y in value.points),
            endpoints=tuple(_transform(p, action) for p in value.endpoints),
        )
    raise AssertionError(f"no {action} transform for {type(value).__name__}")


def _semantic_signature(value: Any) -> Any:
    if isinstance(value, Real):
        return float(value)
    if isinstance(value, L.Scene):
        return len(value.objects)
    if isinstance(value, L.ObjectMask):
        return value.object_id
    if isinstance(value, CircleIntersectionWitness):
        return (value.relation, len(value.points), value.confidence, value.residual)
    if isinstance(value, CirclePairWitness):
        return (value.confidence, value.residual)
    if isinstance(value, CircleWitness):
        return (value.residual, value.confidence)
    if isinstance(value, ArcWitness):
        return (value.angle_degrees, value.residual, value.confidence)
    if isinstance(value, AngleWitness):
        return (
            value.degrees, value.uncertainty_degrees,
            value.residual, value.confidence,
        )
    if isinstance(value, PointContactSignature):
        return (
            tuple(gap.degrees for gap in value.exterior_gaps),
            tuple(gap.uncertainty_degrees for gap in value.exterior_gaps),
            value.residual,
            value.confidence,
        )
    if isinstance(value, PolygonWitness):
        return (value.side_count, value.residual, value.confidence)
    if isinstance(value, SkeletonGraphWitness):
        return (value.endpoint_count, value.branch_count, value.cycle_count)
    if isinstance(value, PartGraphWitness):
        return (len(value.parts), len(value.contacts), len(value.adjacency))
    if isinstance(value, PartWitness):
        return (value.part_id, value.role,
                len(value.contour.points) if value.contour else 0)
    if isinstance(value, ContactWitness):
        return (value.relation, len(value.points), value.confidence)
    if isinstance(value, ContourWitness):
        return (value.is_closed, value.confidence)
    if isinstance(value, CurveWitness):
        return (len(value.points), len(value.endpoints), value.confidence)
    if isinstance(value, RadialArrangementWitness):
        return (value.part_count, value.symmetry_order,
                value.confidence, value.residual)
    raise AssertionError(f"no semantic signature for {type(value).__name__}")


INVARIANCE_CASES = tuple(
    (contract.name, action)
    for contract in REGISTRY.contracts()
    for action in sorted(contract.invariances)
)

# These are observed contract violations, not tolerances.  Strict xfail keeps
# the debt visible and deliberately fails with XPASS as soon as an
# implementation starts satisfying its declaration.
KNOWN_INVARIANCE_VIOLATIONS = {}

INVARIANCE_PARAMS = tuple(
    pytest.param(
        contract_name,
        action,
        id=f"{contract_name}-{action}",
        marks=(pytest.mark.xfail(
            strict=True,
            reason=KNOWN_INVARIANCE_VIOLATIONS[(contract_name, action)],
        ) if (contract_name, action) in KNOWN_INVARIANCE_VIOLATIONS else ()),
    )
    for contract_name, action in INVARIANCE_CASES
)


def test_invariance_matrix_covers_every_claimed_action_exactly_once():
    expected = {
        (contract.name, action)
        for contract in REGISTRY.contracts()
        for action in contract.invariances
    }
    assert len(INVARIANCE_CASES) == len(set(INVARIANCE_CASES))
    assert set(INVARIANCE_CASES) == expected
    assert set(KNOWN_INVARIANCE_VIOLATIONS) <= expected


@pytest.mark.parametrize(
    ("contract_name", "action"),
    INVARIANCE_PARAMS,
)
def test_each_claimed_invariance_preserves_the_leg_semantics(contract_name, action):
    contract = CONTRACTS[contract_name]
    original = _fixture_for(contract_name)
    transformed = _transform(original, action)
    before = _semantic_signature(contract.implementation(original))
    after = _semantic_signature(contract.implementation(transformed))
    if isinstance(before, float):
        assert after == pytest.approx(before, rel=2e-2, abs=1e-6)
    else:
        assert after == pytest.approx(before, rel=2e-2, abs=1e-6)


EQUIVARIANCE_CASES = tuple(
    (contract.name, action)
    for contract in REGISTRY.contracts()
    for action in sorted(contract.equivariances)
)


def _assert_values_close(actual: Any, expected: Any) -> None:
    if isinstance(expected, np.ndarray):
        assert isinstance(actual, np.ndarray)
        np.testing.assert_array_equal(actual, expected)
        return
    if dataclasses.is_dataclass(expected):
        assert type(actual) is type(expected)
        for item in dataclasses.fields(expected):
            if item.name == "provenance":
                continue
            _assert_values_close(
                getattr(actual, item.name), getattr(expected, item.name))
        return
    if isinstance(expected, Real) and not isinstance(expected, (bool, np.bool_)):
        assert float(actual) == pytest.approx(
            float(expected), rel=2e-2, abs=1e-7)
        return
    if isinstance(expected, (tuple, list)):
        assert type(actual) is type(expected)
        assert len(actual) == len(expected)
        for observed, wanted in zip(actual, expected):
            _assert_values_close(observed, wanted)
        return
    assert actual == expected


def test_equivariance_matrix_covers_every_claimed_action_exactly_once():
    expected = {
        (contract.name, action)
        for contract in REGISTRY.contracts()
        for action in contract.equivariances
    }
    assert len(EQUIVARIANCE_CASES) == len(set(EQUIVARIANCE_CASES))
    assert set(EQUIVARIANCE_CASES) == expected


@pytest.mark.parametrize(
    ("contract_name", "action"), EQUIVARIANCE_CASES,
    ids=[f"{name}-{action}" for name, action in EQUIVARIANCE_CASES],
)
def test_each_claimed_equivariance_transports_the_leg_output(
        contract_name, action):
    contract = CONTRACTS[contract_name]
    original = _fixture_for(contract_name)
    before = contract.implementation(original)
    expected = _transform(before, action)
    after = contract.implementation(_transform(original, action))
    _assert_values_close(after, expected)


def test_constructive_geometry_outputs_transform_equivariantly():
    for action in ("translation", "uniform_scale"):
        factor = 2.0 if action == "uniform_scale" else 1.0

        circle_input = _circle_contour(closed=True)
        before_circle = L.fit_circle(circle_input)
        after_circle = L.fit_circle(_transform(circle_input, action))
        assert (after_circle.center.x, after_circle.center.y) == pytest.approx(
            _xy(before_circle.center.x, before_circle.center.y, action))
        assert after_circle.radius == pytest.approx(before_circle.radius * factor)

        arc_input = _circle_contour(closed=False)
        before_arc = L.fit_arc(arc_input)
        after_arc = L.fit_arc(_transform(arc_input, action))
        assert (after_arc.center.x, after_arc.center.y) == pytest.approx(
            _xy(before_arc.center.x, before_arc.center.y, action))
        assert after_arc.radius == pytest.approx(before_arc.radius * factor)
        assert after_arc.angle_degrees == pytest.approx(before_arc.angle_degrees)

        before_points = L.circle_pair_intersection(_circle_pair()).points
        after_points = L.circle_pair_intersection(
            _transform(_circle_pair(), action)).points
        assert sorted((p.x, p.y) for p in after_points) == pytest.approx(
            sorted(_xy(p.x, p.y, action) for p in before_points))


FAILURE_CASES = (
    ("extract_contours", "empty_component",
     L.ObjectMask(np.zeros((32, 32), dtype=bool), "empty"),
     L.WitnessAbsent),
    ("extract_contours", "not_simple_curve", _stroke_object("cross"),
     L.WitnessAbsent),
    ("fit_polygon", "open_contour", _circle_contour(closed=False),
     L.WitnessAbsent),
    ("fit_polygon", "too_few_sides", _circle_contour(closed=True),
     L.WitnessAbsent),
    ("fit_polygon", "high_residual", _wavy_closed_contour(),
     L.WitnessAbsent),
    ("detect_corners", "open_contour", _circle_contour(closed=False),
     L.WitnessAbsent),
    ("detect_corners", "too_few_sides", _circle_contour(closed=True),
     L.WitnessAbsent),
    ("detect_corners", "high_residual", _wavy_closed_contour(),
     L.WitnessAbsent),
    ("classify_triangle", "wrong_side_count", _polygon(4), L.WitnessAbsent),
    ("classify_quadrilateral", "wrong_side_count", _polygon(3), L.WitnessAbsent),
    ("fit_circle", "not_enough_points",
     ContourWitness(points=((0.0, 0.0),), is_closed=True), L.WitnessAbsent),
    ("fit_circle", "open_contour", _circle_contour(closed=False), L.WitnessAbsent),
    ("fit_circle", "high_residual", _square_contour(), L.WitnessAbsent),
    ("fit_arc", "not_enough_points",
     ContourWitness(points=((0.0, 0.0),)), L.WitnessAbsent),
    ("fit_arc", "closed_contour", _circle_contour(closed=True),
     L.WitnessAbsent),
    ("fit_arc", "degenerate_fit", _line_contour(), L.WitnessAbsent),
    ("fit_arc", "high_residual",
     dataclasses.replace(_square_contour(), is_closed=False),
     L.WitnessAbsent),
    ("fit_arc", "insufficient_angular_support", ContourWitness(
        points=tuple((20.0 * math.cos(t), 20.0 * math.sin(t))
                     for t in np.linspace(0.0, math.pi, 8)),
        is_closed=False,
    ), L.WitnessAbsent),
    ("fit_arc", "direction_reversal", ContourWitness(
        points=tuple((20.0 * math.cos(t), 20.0 * math.sin(t))
                     for t in (0.0, 0.1, 0.2, 0.3, 0.2,
                               0.35, 0.5, 0.65, 0.8)),
        is_closed=False,
    ), L.WitnessAbsent),
    ("fit_line_segment", "not_enough_points", ContourWitness(points=()),
     L.WitnessAbsent),
    ("fit_line_segment", "closed_contour", _square_contour(),
     L.WitnessAbsent),
    ("fit_line_segment", "degenerate_segment", ContourWitness(
        points=((4.0, 4.0), (4.0, 4.0)), is_closed=False),
     L.WitnessAbsent),
    ("fit_line_segment", "high_residual", _circle_contour(closed=False),
     L.WitnessAbsent),
    ("fit_multiple_circles", "fewer_than_two_candidates",
     L.Scene(np.zeros((32, 32), dtype=np.uint8), ()), L.WitnessAbsent),
    ("fit_multiple_circles", "high_residual",
     L.Scene(
         np.logical_or(_stroke_object().mask, np.roll(_stroke_object().mask, 8, axis=0)),
         (_stroke_object(), L.ObjectMask(np.roll(_stroke_object().mask, 8, axis=0), "square_2")),
     ), L.WitnessAbsent),
    ("detect_contact", "no_contact", PartGraphWitness(), L.WitnessAbsent),
    ("detect_attachment", "no_attachment", PartGraphWitness(), L.WitnessAbsent),
    ("detect_intersection", "no_crossing", _contact_graph(), L.WitnessAbsent),
    ("minimum_incident_angle", "no_junction", PartGraphWitness(),
     L.WitnessAbsent),
    ("minimum_incident_angle", "insufficient_incident_rays",
     PartGraphWitness(contacts=(ContactWitness(
         source_a="missing-a", source_b="missing-b",
         points=(PointWitness(x=0.0, y=0.0),),
     ),)), L.WitnessIndeterminate),
    ("minimum_incident_angle", "high_residual",
     _high_residual_junction_graph(), L.WitnessIndeterminate),
    ("extract_point_contact_signature", "no_point_contact_signature",
     np.zeros((64, 64), dtype=np.uint8), L.WitnessAbsent),
    ("circle_pair_intersection", "no_intersection", CirclePairWitness(
        first=_circle("a", 0.0, 0.0, 2.0),
        second=_circle("b", 10.0, 0.0, 2.0),
        center_distance=10.0,
    ), L.WitnessAbsent),
    ("detect_radial_arrangement", "fewer_than_three_parts",
     PartGraphWitness(parts=(_part("only", 1, 1),)),
     L.WitnessAbsent),
    ("detect_radial_arrangement", "no_shared_hub",
     PartGraphWitness(parts=tuple(
         _part(f"apart_{i}", 10.0 + i * 20.0, 20.0)
         for i in range(3))),
     L.WitnessAbsent),
    ("detect_radial_arrangement", "poor_radial_fit", PartGraphWitness(
        parts=(_part("a", 55.0, 40.0), _part("b", 65.0, 40.0),
               _part("c", 40.0, 55.0)),
        contacts=(ContactWitness(
            source_a="a", source_b="b",
            points=(PointWitness(x=40.0, y=40.0, source_id="hub"),),
            relation="attachment",
        ),),
        adjacency=(("a", "b"), ("a", "c"), ("b", "c")),
    ), L.WitnessAbsent),
    ("select_largest", "no_objects",
     L.Scene(np.zeros((32, 32), dtype=np.uint8), ()), L.WitnessAbsent),
    ("select_largest_object", "no_objects",
     L.Scene(np.zeros((32, 32), dtype=np.uint8), ()), L.WitnessAbsent),
    ("select_smallest_object", "no_objects",
     L.Scene(np.zeros((32, 32), dtype=np.uint8), ()), L.WitnessAbsent),
    ("select_largest_part", "no_parts", PartGraphWitness(), L.WitnessAbsent),
)


def test_adversarial_matrix_covers_every_contract_that_advertises_failure():
    advertised = {
        (name, mode)
        for name, contract in CONTRACTS.items()
        for mode in contract.failure_modes
    }
    semantic_absence_cases = {
        (name, mode)
        for name, mode, _value, exc in FAILURE_CASES
        if exc is L.WitnessAbsent
    }
    assert semantic_absence_cases == advertised
    assert {
        (name, mode)
        for name, mode, _value, exc in FAILURE_CASES
        if exc is L.WitnessIndeterminate
    } <= {
        (name, mode)
        for name, contract in CONTRACTS.items()
        for mode in contract.indeterminate_modes
    }


@pytest.mark.parametrize(
    ("contract_name", "failure_mode", "value", "exception_type"),
    FAILURE_CASES,
    ids=[f"{case[0]}:{case[1]}" for case in FAILURE_CASES],
)
def test_advertised_failure_paths_refuse_to_fabricate_witnesses(
        contract_name, failure_mode, value, exception_type):
    advertised_modes = (
        CONTRACTS[contract_name].failure_modes
        if exception_type is L.WitnessAbsent
        else CONTRACTS[contract_name].indeterminate_modes
    )
    assert failure_mode in advertised_modes
    with pytest.raises(exception_type) as raised:
        CONTRACTS[contract_name].implementation(value)
    if exception_type in {L.WitnessAbsent, L.WitnessIndeterminate}:
        assert raised.value.failure_mode == failure_mode


def test_selection_and_measurement_family_has_constructive_meaning():
    scene = L.parse_scene(_panel_with_two_blobs())
    assert L.object_count(scene) == 2.0
    assert L.select_largest(scene).mask.sum() > L.select_smallest_object(scene).mask.sum()
    assert L.select_largest_object(scene) == L.select_largest(scene)
    assert L.select_principal_objects(scene).objects == scene.objects
    assert L.select_all_objects(scene) is scene
    assert L.total_ink(scene.panel) == float(scene.panel.sum())
    assert L.largest_ink(scene) == float(L.select_largest(scene).mask.sum())

    graph = _contact_graph(intersection=True)
    assert L.part_count(graph) == 2.0
    assert L.contact_count(graph) == 1.0
    assert L.intersection_count(graph) == 1.0
    assert L.detect_contact(graph).points
    assert L.detect_intersection(graph).relation == "intersection"

    reversed_scene = L.Scene(scene.panel, tuple(reversed(scene.objects)))
    assert L.select_largest(reversed_scene).mask.sum() == max(
        obj.mask.sum() for obj in scene.objects)
    assert L.select_smallest_object(reversed_scene).mask.sum() == min(
        obj.mask.sum() for obj in scene.objects)
    assert L.select_principal_objects(reversed_scene).objects[0].mask.sum() \
        == L.select_largest(reversed_scene).mask.sum()
    assert L.largest_ink(reversed_scene) \
        == float(L.select_largest(reversed_scene).mask.sum())


def test_total_ink_counts_foreground_support_not_pixel_intensity():
    support = _panel_with_two_blobs().astype(bool)
    dim = support.astype(np.uint8)
    bright = support.astype(np.uint8) * 255
    assert L.total_ink(dim) == L.total_ink(bright) == float(support.sum())


def test_geometric_fitters_reject_sparse_or_localized_false_witnesses():
    tiny_v = ContourWitness(
        source_component_id="tiny_v",
        points=((0.0, 0.0), (1.0, 1.0), (2.0, 0.0)),
        is_closed=False,
    )
    with pytest.raises(L.WitnessAbsent, match="eight") as arc_failure:
        L.fit_arc(tiny_v)
    assert arc_failure.value.failure_mode == "not_enough_points"

    kink = ContourWitness(
        source_component_id="localized_kink",
        points=tuple(
            (float(i), 8.0 if i == 50 else 0.0) for i in range(101)),
        is_closed=False,
    )
    with pytest.raises(L.WitnessAbsent) as line_failure:
        L.fit_line_segment(kink)
    assert line_failure.value.failure_mode == "high_residual"

    theta = np.linspace(0.0, 2.0 * math.pi, 256, endpoint=False)
    radius = 30.0 + 4.0 * np.cos(4.0 * theta)
    smooth_lobes = ContourWitness(
        source_component_id="smooth_lobes",
        points=tuple(zip(64.0 + radius * np.cos(theta),
                         64.0 + radius * np.sin(theta))),
        is_closed=True,
    )
    with pytest.raises(L.WitnessAbsent) as polygon_failure:
        L.fit_polygon(smooth_lobes)
    assert polygon_failure.value.failure_mode == "high_residual"


@pytest.mark.parametrize("scale", (0.5, 1.0, 2.0))
def test_circle_fit_rejects_a_short_deep_dent_hidden_by_global_rms(scale):
    theta = np.linspace(0.0, 2.0 * math.pi, 360, endpoint=False)
    angular_distance = np.minimum(theta, 2.0 * math.pi - theta)
    half_width = math.radians(15.0)
    dent = 6.0 * scale * np.maximum(
        0.0, 1.0 - angular_distance / half_width)
    radius = 30.0 * scale - dent
    contour = ContourWitness(
        source_component_id="dented_circle",
        points=tuple(zip(
            64.0 * scale + radius * np.cos(theta),
            64.0 * scale + radius * np.sin(theta),
        )),
        is_closed=True,
    )

    raw = L._fit_circle_raw(contour)
    radial_q95, radial_max = L._radial_deviation_tail(contour, raw)
    assert raw.residual < L.MAX_CIRCLE_RESIDUAL
    assert (radial_q95 > L.MAX_CIRCLE_RADIAL_Q95
            or radial_max > L.MAX_CIRCLE_RADIAL_MAX)
    with pytest.raises(L.WitnessAbsent, match="radial residuals") as failure:
        L.fit_circle(contour)
    assert failure.value.failure_mode == "high_residual"


@pytest.mark.parametrize("scale", (0.5, 1.0, 2.0))
def test_arc_fit_rejects_a_short_deep_dent_hidden_by_global_rms(scale):
    theta = np.linspace(0.0, math.pi, 181)
    half_width = math.radians(10.0)
    dent = 12.0 * scale * np.maximum(
        0.0, 1.0 - np.abs(theta - math.pi / 2.0) / half_width)
    radius = 38.0 * scale - dent
    contour = ContourWitness(
        source_component_id="dented_arc",
        points=tuple(zip(
            64.0 * scale + radius * np.cos(theta),
            64.0 * scale + radius * np.sin(theta),
        )),
        is_closed=False,
    )

    raw = L._fit_circle_raw(contour)
    radial_q95, radial_max = L._radial_deviation_tail(contour, raw)
    assert raw.residual < L.MAX_ARC_RESIDUAL
    assert (radial_q95 > L.MAX_ARC_RADIAL_Q95
            or radial_max > L.MAX_ARC_RADIAL_MAX)
    with pytest.raises(L.WitnessAbsent, match="radial residuals") as failure:
        L.fit_arc(contour)
    assert failure.value.failure_mode == "high_residual"


def test_multi_circle_fit_scans_past_non_circle_distractors():
    circle_scene = _two_circle_scene()
    distractors = []
    for index in range(6):
        mask = np.zeros_like(circle_scene.panel, dtype=bool)
        y = 4 + 7 * index
        mask[y:y + 3, 104:109] = True
        distractors.append(L.ObjectMask(mask, f"distractor_{index}"))
    panel = np.logical_or.reduce([
        *(obj.mask for obj in distractors),
        *(obj.mask for obj in circle_scene.objects),
    ]).astype(np.uint8)
    scene = L.Scene(panel, tuple(distractors) + circle_scene.objects)

    pair = L.fit_multiple_circles(scene)
    assert {pair.first.source_component_id,
            pair.second.source_component_id} == {"circle_0", "circle_1"}


@pytest.mark.parametrize("kind", ("square", "cross"))
def test_hough_proposals_do_not_turn_non_circular_strokes_into_circles(kind):
    obj = _stroke_object(kind)
    scene = L.Scene(obj.mask.astype(np.uint8), (obj,))
    with pytest.raises(L.WitnessAbsent) as failure:
        L.fit_multiple_circles(scene)
    assert failure.value.failure_mode in {
        "fewer_than_two_candidates", "high_residual",
    }


def test_binarize_is_threshold_semantics_not_uint8_wraparound():
    panel = np.array([[-3.0, 0.0, 0.01, 2.0, 300.0]])
    result = L.binarize_panel(panel)
    assert result.tolist() == [[0, 0, 1, 1, 1]]
    assert result.dtype == np.uint8


def test_thinning_cache_is_shape_aware_and_order_independent():
    wide = np.array([[1, 1, 1], [0, 0, 0]], dtype=np.uint8)
    tall = np.array([[1, 1], [1, 0], [0, 0]], dtype=np.uint8)
    assert wide.tobytes() == tall.tobytes()

    L._THIN_CACHE.clear()
    wide_first = L._thinned(wide).copy()
    tall_second = L._thinned(tall).copy()
    L._THIN_CACHE.clear()
    tall_first = L._thinned(tall).copy()
    wide_second = L._thinned(wide).copy()

    assert wide_first.shape == wide_second.shape == wide.shape
    assert tall_first.shape == tall_second.shape == tall.shape
    np.testing.assert_array_equal(wide_first, wide_second)
    np.testing.assert_array_equal(tall_first, tall_second)


def test_short_real_junction_arms_survive_part_decomposition():
    mask = np.zeros((30, 30), dtype=bool)
    mask[5:25, 15] = True
    mask[10, 12:19] = True

    def graph_at_scale(factor: int):
        scaled = np.repeat(np.repeat(mask, factor, axis=0), factor, axis=1)
        return L.decompose_component_into_parts(
            L.ObjectMask(scaled, f"short_cross_{factor}"))

    for factor in (1, 2):
        graph = graph_at_scale(factor)
        assert len(graph.parts) == 4
        assert len(graph.contacts) == 1
        assert graph.contacts[0].relation == "intersection"
        assert len(graph.adjacency) == 6


def test_loop_reentry_counts_incident_half_edges_not_only_components():
    def stroke(polylines):
        panel = np.zeros((128, 128), dtype=np.uint8)
        for points in polylines:
            from dataset import _draw_polyline
            _draw_polyline(panel, np.asarray(points, dtype=float))
        return L.select_largest(L.parse_scene(panel))

    figure_eight = stroke([[
        (64, 64), (40, 40), (20, 64), (40, 88), (64, 64),
        (88, 40), (108, 64), (88, 88), (64, 64),
    ]])
    lollipop = stroke((
        ((40, 40), (80, 40), (80, 80), (40, 80), (40, 40)),
        ((60, 80), (60, 110)),
    ))

    crossing = L.decompose_component_into_parts(figure_eight)
    attachment = L.decompose_component_into_parts(lollipop)
    assert len(crossing.parts) == 2
    assert [contact.relation for contact in crossing.contacts] \
        == ["intersection"]
    assert len(attachment.parts) == 2
    assert [contact.relation for contact in attachment.contacts] \
        == ["attachment"]


def test_offgrid_raster_rotation_is_not_overclaimed_by_axis_metrics():
    from scipy import ndimage

    rectangle = np.zeros((128, 128), dtype=np.uint8)
    rectangle[54:74, 25:103] = 1
    line = np.zeros_like(rectangle)
    line[64, 20:108] = 1

    def rotated_object(panel):
        rotated = ndimage.rotate(
            panel.astype(float), 45.0, order=1, reshape=False)
        return L.select_largest(L.parse_scene((rotated > 0.25).astype(np.uint8)))

    rectangle_obj = L.select_largest(L.parse_scene(rectangle))
    line_obj = L.select_largest(L.parse_scene(line))
    assert L.bbox_occupancy(rotated_object(rectangle)) \
        != pytest.approx(L.bbox_occupancy(rectangle_obj), rel=0.1)
    assert L.elongation(rotated_object(line)) \
        != pytest.approx(L.elongation(line_obj), rel=0.1)
    assert "rotation" not in CONTRACTS["bbox_occupancy"].invariances
    assert "rotation" not in CONTRACTS["elongation"].invariances


def _raster_curve_contour(
        points: np.ndarray, angle: float, stroke_width: int) -> ContourWitness:
    from dataset import _draw_polyline
    from scipy import ndimage

    panel = np.zeros((128, 128), dtype=np.uint8)
    _draw_polyline(panel, np.asarray(points, dtype=float))
    raster = ndimage.rotate(
        panel.astype(float), angle, order=1, reshape=False) > 0.25
    if stroke_width > 1:
        raster = ndimage.binary_dilation(
            raster, iterations=(stroke_width - 1) // 2)
    return L.extract_contours(L.select_largest(
        L.parse_scene(raster.astype(np.uint8))))


def test_raster_curve_counts_are_coherent_across_angle_and_stroke_width():
    line = np.array(((20.0, 64.0), (108.0, 64.0)))
    arc_theta = np.linspace(-0.75 * math.pi, 0.75 * math.pi, 150)
    arc = np.column_stack((
        64.0 + 35.0 * np.cos(arc_theta),
        64.0 + 35.0 * np.sin(arc_theta),
    ))
    wave_x = np.linspace(20.0, 108.0, 180)
    wave = np.column_stack((
        wave_x,
        64.0 + 22.0 * np.sin(3.0 * math.pi * (wave_x - 20.0) / 88.0),
    ))

    for angle in range(0, 90, 5):
        for stroke_width in (1, 3, 5, 7):
            label = f"angle={angle}, width={stroke_width}"
            line_contour = _raster_curve_contour(
                line, angle, stroke_width)
            L.fit_line_segment(line_contour)
            assert L.count_inflections(line_contour) == 0.0, label
            assert L.count_curve_parts(line_contour) == 1.0, label

            arc_contour = _raster_curve_contour(
                arc, angle, stroke_width)
            L.fit_arc(arc_contour)
            assert L.count_inflections(arc_contour) == 0.0, label
            assert L.count_curve_parts(arc_contour) == 1.0, label

            wave_contour = _raster_curve_contour(
                wave, angle, stroke_width)
            assert L.count_inflections(wave_contour) == 2.0, label
            assert L.count_curve_parts(wave_contour) == 3.0, label


def test_inflection_counter_has_a_nonzero_closed_curve_fixture():
    assert L.count_inflections(_wavy_closed_contour()) == 14.0


def test_acute_triangle_chain_survives_offgrid_angle_and_stroke_width():
    triangle = np.asarray((
        (64.0, 22.0), (105.0, 98.0),
        (23.0, 98.0), (64.0, 22.0),
    ))
    for angle in range(0, 90, 5):
        for stroke_width in (1, 3, 5, 7):
            label = f"angle={angle}, width={stroke_width}"
            contour = _raster_curve_contour(
                triangle, angle, stroke_width)
            assert contour.is_closed, label
            polygon = L.fit_polygon(contour)
            assert polygon.side_count == 3, label
            assert isinstance(L.classify_triangle(polygon), TriangleWitness)


def test_closed_contour_regularization_does_not_hide_real_branches():
    from dataset import _draw_polyline

    def object_for(polylines):
        panel = np.zeros((128, 128), dtype=np.uint8)
        for points in polylines:
            _draw_polyline(panel, np.asarray(points, dtype=float))
        return L.select_largest(L.parse_scene(panel))

    figure_eight = object_for(((
        (64, 64), (40, 40), (20, 64), (40, 88), (64, 64),
        (88, 40), (108, 64), (88, 88), (64, 64),
    ),))
    lollipop = object_for((
        ((40, 40), (80, 40), (80, 80), (40, 80), (40, 40)),
        ((60, 80), (60, 110)),
    ))
    for obj in (figure_eight, lollipop):
        with pytest.raises(L.WitnessAbsent) as failure:
            L.extract_contours(obj)
        assert failure.value.failure_mode == "not_simple_curve"


def test_t_junction_topology_survives_offgrid_angle_and_stroke_width():
    from dataset import _draw_polyline
    from scipy import ndimage

    panel = np.zeros((128, 128), dtype=np.uint8)
    _draw_polyline(panel, np.asarray(((20, 42), (108, 42)), dtype=float))
    _draw_polyline(panel, np.asarray(((64, 42), (64, 102)), dtype=float))
    for angle in range(0, 90, 5):
        for stroke_width in (1, 3, 5, 7):
            label = f"angle={angle}, width={stroke_width}"
            raster = ndimage.rotate(
                panel.astype(float), angle, order=1, reshape=False) > 0.25
            if stroke_width > 1:
                raster = ndimage.binary_dilation(
                    raster, iterations=(stroke_width - 1) // 2)
            obj = L.select_largest(L.parse_scene(
                raster.astype(np.uint8)))
            skeleton = L.build_skeleton_graph(obj)
            assert (skeleton.endpoint_count, skeleton.branch_count,
                    skeleton.cycle_count) == (3, 1, 0), label
            graph = L.build_part_graph(obj)
            assert len(graph.parts) == 3, label
            assert len(graph.contacts) == 1, label
            assert graph.contacts[0].relation == "attachment", label
            with pytest.raises(L.WitnessAbsent) as failure:
                L.extract_contours(obj)
            assert failure.value.failure_mode == "not_simple_curve"


def test_raster_radial_chain_survives_offgrid_angle_and_stroke_width():
    from dataset import _draw_polyline
    from scipy import ndimage

    center = np.asarray((64.0, 64.0))
    panel = np.zeros((128, 128), dtype=np.uint8)
    for index in range(3):
        theta = 0.13 + 2.0 * math.pi * index / 3.0
        endpoint = center + 42.0 * np.asarray((
            math.cos(theta), math.sin(theta)))
        _draw_polyline(panel, np.stack((center, endpoint)))
    for angle in range(0, 90, 5):
        for stroke_width in (1, 3, 5, 7):
            label = f"angle={angle}, width={stroke_width}"
            raster = ndimage.rotate(
                panel.astype(float), angle, order=1, reshape=False) > 0.25
            if stroke_width > 1:
                raster = ndimage.binary_dilation(
                    raster, iterations=(stroke_width - 1) // 2)
            graph = L.build_part_graph(L.select_largest(
                L.parse_scene(raster.astype(np.uint8))))
            assert len(graph.parts) == 3, label
            assert len(graph.contacts) == 1, label
            assert graph.contacts[0].relation == "attachment", label
            radial = L.detect_radial_arrangement(graph)
            assert radial.part_count == 3, label
            assert radial.symmetry_order == 0, label
            assert radial.confidence >= 0.72, label


def test_circle_and_quadrilateral_chains_survive_raster_perturbations():
    circle_theta = np.linspace(0.0, 2.0 * math.pi, 240)
    circle = np.column_stack((
        64.0 + 32.0 * np.cos(circle_theta),
        64.0 + 32.0 * np.sin(circle_theta),
    ))
    square = np.asarray((
        (28.0, 28.0), (100.0, 28.0), (100.0, 100.0),
        (28.0, 100.0), (28.0, 28.0),
    ))
    for angle in range(0, 90, 5):
        for stroke_width in (1, 3, 5, 7):
            label = f"angle={angle}, width={stroke_width}"
            circle_contour = _raster_curve_contour(
                circle, angle, stroke_width)
            fitted_circle = L.fit_circle(circle_contour)
            assert fitted_circle.residual <= L.MAX_CIRCLE_RESIDUAL, label

            square_contour = _raster_curve_contour(
                square, angle, stroke_width)
            quadrilateral = L.classify_quadrilateral(
                L.fit_polygon(square_contour))
            assert quadrilateral.side_count == 4, label


def test_cross_intersection_chain_survives_raster_perturbations():
    from dataset import _draw_polyline
    from scipy import ndimage

    panel = np.zeros((128, 128), dtype=np.uint8)
    _draw_polyline(panel, np.asarray(((20, 64), (108, 64)), dtype=float))
    _draw_polyline(panel, np.asarray(((64, 20), (64, 108)), dtype=float))
    for angle in range(0, 90, 5):
        for stroke_width in (1, 3, 5, 7):
            label = f"angle={angle}, width={stroke_width}"
            raster = ndimage.rotate(
                panel.astype(float), angle, order=1, reshape=False) > 0.25
            if stroke_width > 1:
                raster = ndimage.binary_dilation(
                    raster, iterations=(stroke_width - 1) // 2)
            obj = L.select_largest(L.parse_scene(
                raster.astype(np.uint8)))
            skeleton = L.build_skeleton_graph(obj)
            assert (skeleton.endpoint_count, skeleton.branch_count,
                    skeleton.cycle_count) == (4, 1, 0), label
            graph = L.build_part_graph(obj)
            assert len(graph.parts) == 4, label
            assert len(graph.contacts) == 1, label
            intersection = L.detect_intersection(graph)
            assert intersection.relation == "intersection", label

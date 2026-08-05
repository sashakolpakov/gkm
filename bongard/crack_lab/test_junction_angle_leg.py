"""Focused regressions for typed local junction-angle extraction."""
from __future__ import annotations

import dataclasses
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import semantic_legs as L
from visual_witnesses import (
    ContactWitness,
    ContourWitness,
    IntersectionWitness,
    PartGraphWitness,
    PartWitness,
    PointWitness,
)


def _polar(angle_degrees: float, radius: float) -> tuple[float, float]:
    angle = math.radians(angle_degrees)
    return radius * math.cos(angle), radius * math.sin(angle)


def _two_ray_part(
        part_id: str, first_angle: float, second_angle: float,
        ) -> PartWitness:
    first = [_polar(first_angle, radius) for radius in np.linspace(1.0, 30.0, 30)]
    outer = [
        _polar(angle, 30.0)
        for angle in np.linspace(first_angle, second_angle, 61)[1:-1]
    ]
    second = [
        _polar(second_angle, radius)
        for radius in np.linspace(30.0, 1.0, 30)
    ]
    contour = ContourWitness(
        source_component_id=part_id,
        points=tuple(first + outer + second),
        is_closed=False,
        confidence=0.98,
        provenance=("synthetic-two-ray-part",),
    )
    return PartWitness(
        part_id=part_id,
        role="stroke-loop",
        source_component_id=part_id,
        contour=contour,
        confidence=0.98,
        provenance=("synthetic-part",),
    )


def _junction_graph_from_pairs(
        angles: tuple[tuple[float, float], tuple[float, float]],
        ) -> PartGraphWitness:
    parts = tuple(
        _two_ray_part(f"part-{index}", *pair)
        for index, pair in enumerate(angles)
    )
    junction = PointWitness(x=0.0, y=0.0, source_id="junction")
    contact = IntersectionWitness(
        source_a=parts[0].part_id,
        source_b=parts[1].part_id,
        points=(junction,),
        relation="intersection",
        confidence=0.97,
        provenance=("synthetic-junction",),
    )
    return PartGraphWitness(
        parts=parts,
        contacts=(contact,),
        adjacency=((parts[0].part_id, parts[1].part_id),),
        confidence=0.96,
        provenance=("synthetic-graph",),
    )


def junction_graph(minimum_degrees: float) -> PartGraphWitness:
    if minimum_degrees == 30.0:
        angles = ((0.0, 140.0), (30.0, 250.0))
    elif minimum_degrees == 60.0:
        angles = ((0.0, 160.0), (60.0, 260.0))
    else:  # pragma: no cover - the fixture intentionally has two contracts
        raise ValueError("fixture supports only 30 or 60 degrees")
    return _junction_graph_from_pairs(angles)


def _transform_graph(
        graph: PartGraphWitness, matrix: np.ndarray,
        shift: tuple[float, float],
        ) -> PartGraphWitness:
    offset = np.asarray(shift, dtype=float)

    def xy(x: float, y: float) -> tuple[float, float]:
        transformed = matrix @ np.asarray((x, y), dtype=float) + offset
        return float(transformed[0]), float(transformed[1])

    def point(value: PointWitness) -> PointWitness:
        x, y = xy(value.x, value.y)
        return dataclasses.replace(value, x=x, y=y)

    def part(value: PartWitness) -> PartWitness:
        assert value.contour is not None
        contour = dataclasses.replace(
            value.contour,
            points=tuple(xy(x, y) for x, y in value.contour.points),
        )
        return dataclasses.replace(value, contour=contour)

    def contact(value: ContactWitness) -> ContactWitness:
        return dataclasses.replace(
            value, points=tuple(point(item) for item in value.points))

    return dataclasses.replace(
        graph,
        parts=tuple(part(item) for item in graph.parts),
        contacts=tuple(contact(item) for item in graph.contacts),
    )


def test_minimum_incident_angle_preserves_raw_30_vs_60_difference() -> None:
    acute = L.minimum_incident_angle(junction_graph(30.0))
    wider = L.minimum_incident_angle(junction_graph(60.0))

    assert acute.degrees == pytest.approx(30.0, abs=1e-9)
    assert wider.degrees == pytest.approx(60.0, abs=1e-9)
    assert L.angle_degrees(acute) == pytest.approx(30.0, abs=1e-9)
    assert L.angle_degrees(wider) == pytest.approx(60.0, abs=1e-9)
    # The old obliqueness projection is intentionally symmetric around 45°
    # and therefore cannot replace raw magnitude for this contrast.
    assert L.angle_noncardinality_degrees(acute) \
        == pytest.approx(L.angle_noncardinality_degrees(wider), abs=1e-9)


def test_minimum_incident_angle_never_selects_a_same_part_corner() -> None:
    # part-0 contains a 10-degree interior gap.  The closest ray belonging to
    # the contact-bound *other* part is 30 degrees away (10 -> 40).  Flattening
    # the four rays before minimization would incorrectly return 10.
    graph = _junction_graph_from_pairs(((0.0, 10.0), (40.0, 220.0)))
    angle = L.minimum_incident_angle(graph)
    assert angle.degrees == pytest.approx(30.0, abs=1e-9)
    assert angle.source_a.startswith("part-0:")
    assert angle.source_b.startswith("part-1:")


@pytest.mark.parametrize(
    "matrix,shift",
    (
        (np.asarray(((1.0, 0.0), (0.0, 1.0))), (17.0, -9.0)),
        (3.25 * np.eye(2), (0.0, 0.0)),
        (np.asarray(((0.6, -0.8), (0.8, 0.6))), (0.0, 0.0)),
        (np.asarray(((-1.0, 0.0), (0.0, 1.0))), (0.0, 0.0)),
    ),
    ids=("translation", "uniform-scale", "rotation", "reflection"),
)
def test_minimum_incident_angle_is_intrinsic(
        matrix: np.ndarray, shift: tuple[float, float]) -> None:
    graph = junction_graph(30.0)
    before = L.minimum_incident_angle(graph)
    after = L.minimum_incident_angle(_transform_graph(graph, matrix, shift))
    assert after.degrees == pytest.approx(before.degrees, abs=1e-9)
    assert after.uncertainty_degrees == pytest.approx(
        before.uncertainty_degrees, abs=1e-9)
    expected_vertex = matrix @ np.asarray(
        (before.vertex.x, before.vertex.y)) + np.asarray(shift)
    assert (after.vertex.x, after.vertex.y) == pytest.approx(expected_vertex)


def test_default_registry_exposes_the_complete_typed_chain() -> None:
    registry = L.default_registry()
    extractor = registry.get("minimum_incident_angle")
    measurement = registry.get("angle_degrees")
    assert extractor.domain == ("PartGraphWitness",)
    assert extractor.codomain == "AngleWitness"
    assert measurement.domain == ("AngleWitness",)
    assert measurement.codomain == "Measurement"
    assert extractor.equivariances == frozenset({
        "translation", "uniform_scale", "rotation", "reflection"})
    assert extractor.failure_modes == ("no_junction",)
    assert set(extractor.indeterminate_modes) == {
        "insufficient_incident_rays", "high_residual"}
    assert measurement.invariances == frozenset({
        "translation", "uniform_scale", "rotation", "reflection"})


def test_junction_angle_failures_are_typed() -> None:
    with pytest.raises(L.WitnessAbsent) as absent:
        L.minimum_incident_angle(PartGraphWitness())
    assert absent.value.failure_mode == "no_junction"

    unresolved = PartGraphWitness(
        contacts=(ContactWitness(
            source_a="missing-a",
            source_b="missing-b",
            points=(PointWitness(x=0.0, y=0.0),),
        ),),
    )
    with pytest.raises(L.WitnessIndeterminate) as insufficient:
        L.minimum_incident_angle(unresolved)
    assert insufficient.value.failure_mode == "insufficient_incident_rays"


def test_smoke_problem_recovers_local_turn_near_miss() -> None:
    directory = Path(__file__).with_name("semantic_soft_runs") \
        / "smoke_20260805" / "workspace" / "problem_00"
    if not directory.is_dir():
        pytest.skip("local immutable semantic-soft smoke fixture is unavailable")

    positives = []
    for index in range(6):
        panel = np.load(directory / f"pos_{index}.npy", allow_pickle=False)
        graph = L.build_part_graph(L.parse_scene(panel))
        positives.append(L.angle_degrees(L.minimum_incident_angle(graph)))
    negative = np.load(directory / "neg_4.npy", allow_pickle=False)
    negative_angle = L.angle_degrees(L.minimum_incident_angle(
        L.build_part_graph(L.parse_scene(negative))))

    assert positives == pytest.approx(
        [28.6506, 28.0647, 27.8764, 29.8566, 31.6166, 26.9069],
        abs=0.02,
    )
    assert negative_angle == pytest.approx(59.2012, abs=0.02)
    assert max(positives) + 20.0 < negative_angle

from __future__ import annotations

from dataclasses import replace
from io import BytesIO
import math

from PIL import Image, ImageDraw
import pytest

from bongard.evidence import Disposition
from bongard.loop_geometry import (
    IntInterval,
    LoopGeometryWitness,
    boundary_cycles_for_mask,
)
from bongard.loop_scene_witnesses import extract_loop_scene_witnesses


def _png(polygons: list[list[tuple[float, float]]], *, size: int = 128, width: int = 4) -> bytes:
    image = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(image)
    for polygon in polygons:
        draw.line(polygon + [polygon[0]], fill="black", width=width, joint="curve")
    output = BytesIO()
    image.save(output, format="PNG", optimize=False)
    return output.getvalue()


def _regular_polygon(
    sides: int,
    *,
    center: tuple[float, float] = (64.0, 64.0),
    radius: float = 40.0,
    angle: float = 0.0,
) -> list[tuple[float, float]]:
    return [
        (
            center[0] + radius * math.cos(angle + 2 * math.pi * index / sides),
            center[1] + radius * math.sin(angle + 2 * math.pi * index / sides),
        )
        for index in range(sides)
    ]


def _largest_loop(png_bytes: bytes) -> tuple[LoopGeometryWitness, ...]:
    packet = extract_loop_scene_witnesses(png_bytes)
    return tuple(max(scenario.loops, key=lambda item: item.area_pixels) for scenario in packet.scenarios)


@pytest.mark.parametrize("sides", [3, 4])
@pytest.mark.parametrize("angle", [0.0, 0.31, 0.79, 1.27])
@pytest.mark.parametrize("width", [2, 4, 7])
def test_polygon_side_count_survives_rotation_and_stroke_width(
    sides: int, angle: float, width: int
) -> None:
    loops = _largest_loop(_png([_regular_polygon(sides, angle=angle)], width=width))

    for loop in loops:
        assert loop.substantiveness.disposition is Disposition.PRESENT
        assert loop.polygon.disposition is Disposition.PRESENT
        assert loop.polygon.side_count == IntInterval.point(sides)
        assert loop.edge_obliqueness.disposition is Disposition.PRESENT
        assert LoopGeometryWitness.from_data(loop.to_data()) == loop


def test_circle_fit_is_indeterminate_not_certified_absence() -> None:
    image = Image.new("RGB", (128, 128), "white")
    ImageDraw.Draw(image).ellipse((20, 20, 108, 108), outline="black", width=4)
    output = BytesIO()
    image.save(output, format="PNG", optimize=False)

    for loop in _largest_loop(output.getvalue()):
        assert loop.substantiveness.disposition is Disposition.PRESENT
        assert loop.polygon.disposition is Disposition.INDETERMINATE
        assert loop.polygon.side_count is None
        assert loop.edge_obliqueness.disposition is Disposition.INDETERMINATE


def test_one_admissible_variant_is_not_a_stable_polygon_fit() -> None:
    polygon = [
        (90.176, 90.176),
        (52.913, 105.377),
        (33.401, 72.199),
        (37.733, 37.733),
        (74.811, 23.652),
        (93.426, 56.115),
    ]

    for loop in _largest_loop(_png([polygon], width=3)):
        assert sum(item.admissible for item in loop.polygon.variants) == 1
        assert loop.polygon.disposition is Disposition.INDETERMINATE
        assert loop.polygon.reason_code == "variant_unavailable"
        assert loop.edge_obliqueness.disposition is Disposition.INDETERMINATE


def test_resolution_floor_retains_tiny_hole_but_excludes_it_from_roles() -> None:
    image = Image.new("RGB", (96, 96), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((8, 8, 88, 88), fill="black")
    draw.rectangle((20, 20, 76, 76), fill="white")
    draw.point((12, 12), fill="white")
    output = BytesIO()
    image.save(output, format="PNG", optimize=False)
    packet = extract_loop_scene_witnesses(output.getvalue())

    assert [len(scenario.loops) for scenario in packet.scenarios] == [2, 1, 2]
    for scenario in (packet.scenarios[0], packet.scenarios[2]):
        tiny = min(scenario.loops, key=lambda item: item.area_pixels)
        assert tiny.area_pixels <= 2
        assert tiny.substantiveness.disposition is Disposition.CERTIFIED_ABSENT
        assert tiny.substantiveness.certificate is not None
        assert tiny.polygon.disposition is Disposition.INDETERMINATE


def test_geometry_witness_rejects_forged_side_interval() -> None:
    loop = _largest_loop(_png([_regular_polygon(4, angle=0.37)]))[0]
    assert loop.polygon.side_count == IntInterval.point(4)
    with pytest.raises(ValueError, match="does not envelope"):
        replace(loop.polygon, side_count=IntInterval.point(3))


def test_boundary_stitching_returns_one_closed_cycle_for_simple_region() -> None:
    import numpy as np

    mask = np.zeros((12, 12), dtype=bool)
    mask[2:10, 3:9] = True
    cycles = boundary_cycles_for_mask(mask)

    assert len(cycles) == 1
    assert len(cycles[0]) == 28

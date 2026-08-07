from __future__ import annotations

from copy import deepcopy
from io import BytesIO

from PIL import Image, ImageDraw
import pytest

from bongard.evidence import Disposition
from bongard.loop_scene_witnesses import extract_loop_scene_witnesses
from bongard.triangle_geometry import (
    TriangleClass,
    TriangleGeometryError,
    TriangleGeometryPacket,
    extract_triangle_geometry,
    verify_triangle_geometry,
)


def _panel(points: list[tuple[int, int]], *, tiny: bool = False) -> bytes:
    image = Image.new("RGB", (192, 192), "white")
    draw = ImageDraw.Draw(image)
    draw.line(points + [points[0]], fill="black", width=4, joint="curve")
    if tiny:
        little = [(145, 32), (151, 44), (139, 44)]
        draw.line(little + [little[0]], fill="black", width=3, joint="curve")
    output = BytesIO()
    image.save(output, format="PNG", optimize=False)
    return output.getvalue()


def _largest(packet: TriangleGeometryPacket):
    # loop IDs are scenario-local.  The first raw-scenario loop is enough for
    # the exact synthetic panels, which contain one large carrier.
    return next(
        item
        for item in packet.observations
        if item.object_id.startswith("threshold032.raw/")
    )


def _disposition(packet: TriangleGeometryPacket, kind: TriangleClass):
    return _largest(packet).class_result(kind).disposition


@pytest.mark.parametrize(
    "points, expected, excluded",
    [
        (
            [(35, 145), (96, 39), (157, 145)],
            TriangleClass.EQUILATERAL,
            TriangleClass.RIGHT,
        ),
        (
            [(38, 145), (38, 50), (151, 145)],
            TriangleClass.RIGHT,
            TriangleClass.EQUILATERAL,
        ),
        (
            [(30, 140), (96, 100), (162, 140)],
            TriangleClass.OBTUSE,
            TriangleClass.RIGHT,
        ),
    ],
)
def test_stable_large_triangles_emit_typed_class_evidence(
    points, expected: TriangleClass, excluded: TriangleClass
) -> None:
    png = _panel(points)
    scene = extract_loop_scene_witnesses(png)
    packet = extract_triangle_geometry(scene)

    assert _disposition(packet, expected) is Disposition.PRESENT
    assert _disposition(packet, excluded) in {
        Disposition.CERTIFIED_ABSENT,
        Disposition.INDETERMINATE,
    }
    assert TriangleGeometryPacket.from_data(packet.to_data()) == packet
    assert verify_triangle_geometry(packet, scene) is packet
    assert len(packet.digest) == 64


def test_stable_nontriangle_certifies_triangle_classes_absent() -> None:
    png = _panel([(35, 35), (157, 35), (157, 157), (35, 157)])
    packet = extract_triangle_geometry(extract_loop_scene_witnesses(png))

    observation = _largest(packet)
    assert observation.variants == ()
    assert {
        item.disposition for item in observation.classes
    } == {Disposition.CERTIFIED_ABSENT}


def test_unstable_tiny_polygon_is_indeterminate_never_absent() -> None:
    png = _panel([(35, 145), (96, 39), (157, 145)], tiny=True)
    packet = extract_triangle_geometry(extract_loop_scene_witnesses(png))
    unstable = tuple(
        item
        for item in packet.observations
        if item.polygon_disposition is Disposition.INDETERMINATE
    )

    assert unstable
    assert all(item.variants == () for item in unstable)
    assert all(
        result.disposition is Disposition.INDETERMINATE
        for item in unstable
        for result in item.classes
    )


def test_small_stable_nontriangle_cannot_certify_triangle_absence() -> None:
    image = Image.new("RGB", (192, 192), "white")
    draw = ImageDraw.Draw(image)
    little_square = [(142, 32), (152, 32), (152, 42), (142, 42)]
    draw.line(
        little_square + [little_square[0]],
        fill="black",
        width=3,
        joint="curve",
    )
    output = BytesIO()
    image.save(output, format="PNG", optimize=False)
    packet = extract_triangle_geometry(
        extract_loop_scene_witnesses(output.getvalue())
    )
    guarded = tuple(
        item
        for item in packet.observations
        if item.variants == ()
        and all(
            result.reason_code == "small_loop_below_absence_resolution_guard"
            for result in item.classes
        )
    )
    assert guarded
    assert all(
        result.disposition is Disposition.INDETERMINATE
        for item in guarded
        for result in item.classes
    )


def test_packet_schema_is_candidate_independent_and_strict() -> None:
    packet = extract_triangle_geometry(
        extract_loop_scene_witnesses(
            _panel([(35, 145), (96, 39), (157, 145)])
        )
    )
    encoded = packet.to_data()
    forbidden = {"task", "side", "label", "candidate", "formula", "path", "prose"}

    def keys(value: object) -> set[str]:
        if isinstance(value, dict):
            return set(value).union(*(keys(item) for item in value.values()), set())
        if isinstance(value, list):
            return set().union(*(keys(item) for item in value), set())
        return set()

    assert keys(encoded).isdisjoint(forbidden)
    tampered = deepcopy(encoded)
    tampered["candidate"] = "leak"
    with pytest.raises(TriangleGeometryError, match="fields differ"):
        TriangleGeometryPacket.from_data(tampered)

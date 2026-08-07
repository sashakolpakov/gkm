from __future__ import annotations

from dataclasses import replace
from io import BytesIO

from PIL import Image, ImageDraw
import pytest

from bongard.artifacts import canonical_digest
from bongard.legs.contracts import Unit
from bongard.loop_scene_witnesses import (
    LOOP_SCENE_PACKET,
    LoopScenePacket,
    extract_loop_scene_witnesses,
    loop_scene_fragment,
    verify_loop_scene_packet,
)
from bongard.relational_scene import (
    glue_scene_fragment,
    start_scene_snapshot,
    verify_scene_snapshot,
)
from bongard.visual_witness_bundle import extract_visual_witness_bundle


def _panel(*, move_triangle: bool = False) -> bytes:
    image = Image.new("RGB", (128, 128), "white")
    draw = ImageDraw.Draw(image)
    shift = 8 if move_triangle else 0
    triangle = [(10 + shift, 105), (31 + shift, 65), (51 + shift, 105)]
    quadrilateral = [(58, 30), (108, 20), (115, 94), (65, 104)]
    draw.line(triangle + [triangle[0]], fill="black", width=4, joint="curve")
    draw.line(
        quadrilateral + [quadrilateral[0]], fill="black", width=4, joint="curve"
    )
    output = BytesIO()
    image.save(output, format="PNG", optimize=False)
    return output.getvalue()


def _mixed_fit_panel() -> bytes:
    """One stable triangle and one deliberately non-polygonal closed loop."""

    image = Image.new("RGB", (160, 160), "white")
    draw = ImageDraw.Draw(image)
    triangle = [(15, 130), (40, 70), (65, 130)]
    draw.line(triangle + [triangle[0]], fill="black", width=4, joint="curve")
    draw.ellipse((90, 45, 145, 100), outline="black", width=4)
    output = BytesIO()
    image.save(output, format="PNG", optimize=False)
    return output.getvalue()


def _contains_float(value: object) -> bool:
    if isinstance(value, float):
        return True
    if isinstance(value, list):
        return any(_contains_float(item) for item in value)
    if isinstance(value, dict):
        return any(_contains_float(item) for item in value.values())
    return False


def test_packet_round_trip_enumerates_every_hole_and_serializes_no_floats() -> None:
    packet = extract_loop_scene_witnesses(_panel())

    assert LOOP_SCENE_PACKET.name == "loop_scene_packet"
    assert LoopScenePacket.from_data(packet.to_data()) == packet
    assert not _contains_float(packet.to_data())
    assert len(packet.digest()) == 64
    assert [len(item.loops) for item in packet.scenarios] == [2, 2, 2]
    for scenario in packet.scenarios:
        assert [item.source_hole_id for item in scenario.loops] == [
            "hole-00000000",
            "hole-00000001",
        ]
        assert [item.loop_ids for item in scenario.contacts] == [
            ("loop-00000000", "loop-00000001")
        ]


def test_exact_png_replay_rejects_other_bytes_and_tampering() -> None:
    original = _panel()
    packet = extract_loop_scene_witnesses(original)
    assert verify_loop_scene_packet(packet, expected_png_bytes=original) is packet

    with pytest.raises(ValueError, match="differs from exact PNG replay|parent bundle"):
        verify_loop_scene_packet(packet, expected_png_bytes=_panel(move_triangle=True))
    forged = replace(packet, panel_digest="0" * 64)
    with pytest.raises(ValueError, match="differs from exact PNG replay"):
        verify_loop_scene_packet(forged, expected_png_bytes=original)


def test_packet_attaches_transactionally_to_scene_and_cold_replays() -> None:
    png_bytes = _panel()
    bundle = extract_visual_witness_bundle(png_bytes)
    packet = extract_loop_scene_witnesses(png_bytes)
    parent = start_scene_snapshot(
        bundle, canonical_digest({"schema": "fixture-loop-scene/v1"})
    )
    fragment = loop_scene_fragment(packet, parent)
    child = glue_scene_fragment(parent, fragment)

    assert parent.generation == 0
    assert child.generation == 1
    assert len([item for item in child.entities if item.entity_type == "loop"]) == 6
    assert len(child.facts) == 27
    assert len(
        [item for item in child.facts if item.predicate == "pair.point_contact"]
    ) == 3
    assert (
        verify_scene_snapshot(
            child,
            bundle,
            previous_snapshot=parent,
            applied_fragment=fragment,
        )
        is child
    )


def test_mixed_present_and_unavailable_measurements_keep_declared_units() -> None:
    png_bytes = _mixed_fit_panel()
    bundle = extract_visual_witness_bundle(png_bytes)
    packet = extract_loop_scene_witnesses(png_bytes)
    parent = start_scene_snapshot(
        bundle, canonical_digest({"schema": "fixture-mixed-loop-scene/v1"})
    )
    fragment = loop_scene_fragment(packet, parent)

    side_facts = tuple(
        item for item in fragment.facts if item.predicate == "loop.polygon_side_count"
    )
    oblique_facts = tuple(
        item
        for item in fragment.facts
        if item.predicate == "loop.edge_axis_obliqueness_millidegrees"
    )
    assert {item.unit for item in side_facts} == {Unit.COUNT}
    assert {item.unit for item in oblique_facts} == {Unit.MILLIDEGREES}
    assert any(item.interval is None for item in side_facts)
    assert any(item.interval is not None for item in side_facts)
    assert any(item.interval is None for item in oblique_facts)
    assert any(item.interval is not None for item in oblique_facts)

    child = glue_scene_fragment(parent, fragment)
    assert child.generation == 1


@pytest.mark.parametrize("bad", [b"", b"not-png"])
def test_invalid_input_is_error_not_a_negative(bad: bytes) -> None:
    with pytest.raises((TypeError, ValueError)):
        extract_loop_scene_witnesses(bad)

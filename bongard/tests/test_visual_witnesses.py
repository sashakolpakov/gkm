from __future__ import annotations

from dataclasses import replace
from io import BytesIO

from PIL import Image, ImageDraw
import pytest

from bongard.legs.contracts import ValueType
from bongard.visual_witnesses import (
    VISUAL_WITNESS_CAPABILITY_IDS,
    VISUAL_WITNESS_PACKET,
    VISUAL_WITNESS_SCENARIO_IDS,
    ComponentWitness,
    HoleWitness,
    Q16BBox,
    ScenarioPredicateResult,
    VisualWitnessPacket,
    component_count_by_scenario,
    extract_visual_witnesses,
    owned_hole_count_by_scenario,
    verify_visual_witness_packet,
)


def _panel(*, second_component: bool = True) -> bytes:
    image = Image.new("RGB", (64, 64), "white")
    draw = ImageDraw.Draw(image)
    # A thick rectangular loop: one foreground component and one owned hole.
    draw.rectangle((6, 8, 30, 44), fill="black")
    draw.rectangle((12, 14, 24, 38), fill="white")
    if second_component:
        draw.rectangle((44, 20, 54, 34), fill="black")
    output = BytesIO()
    image.save(output, format="PNG", optimize=False)
    return output.getvalue()


def test_packet_round_trip_and_static_public_vocabulary() -> None:
    packet = extract_visual_witnesses(_panel())

    assert VISUAL_WITNESS_CAPABILITY_IDS == (
        "component.count",
        "hole.owner_count",
    )
    assert VISUAL_WITNESS_SCENARIO_IDS == tuple(
        sorted(VISUAL_WITNESS_SCENARIO_IDS)
    )
    assert VISUAL_WITNESS_PACKET == ValueType("visual_witness_packet")
    assert VisualWitnessPacket.from_data(packet.to_data()) == packet
    assert len(packet.digest()) == 64


def test_exact_component_and_owned_hole_counts_retain_all_scenarios() -> None:
    packet = extract_visual_witnesses(_panel())

    component_result = component_count_by_scenario(packet, 2)
    hole_result = owned_hole_count_by_scenario(packet, 1)

    assert tuple(item.scenario_id for item in packet.scenarios) == (
        VISUAL_WITNESS_SCENARIO_IDS
    )
    assert [item.observed_count for item in component_result.observations] == [2, 2, 2]
    assert [item.matches for item in component_result.observations] == [True] * 3
    assert [item.observed_count for item in hole_result.observations] == [1, 1, 1]
    assert [item.matches for item in hole_result.observations] == [True] * 3
    assert ScenarioPredicateResult.from_data(component_result.to_data()) == component_result

    for scenario in packet.scenarios:
        assert [item.component_id for item in scenario.components] == [
            "component-00000000",
            "component-00000001",
        ]
        assert len(scenario.holes) == 1
        assert scenario.holes[0].owner_component_id == "component-00000000"


def test_exact_png_replay_accepts_original_and_rejects_other_bytes() -> None:
    original = _panel()
    packet = extract_visual_witnesses(original)

    assert verify_visual_witness_packet(packet, expected_png_bytes=original) is packet
    with pytest.raises(ValueError, match="differs from exact PNG replay"):
        verify_visual_witness_packet(
            packet, expected_png_bytes=_panel(second_component=False)
        )


@pytest.mark.parametrize("field", ["bbox", "mask", "owner"])
def test_exact_replay_rejects_geometry_mask_and_owner_forgery(field: str) -> None:
    original = _panel()
    packet = extract_visual_witnesses(original)
    scenario = packet.scenarios[0]
    components = scenario.components
    holes = scenario.holes

    if field == "bbox":
        component = components[0]
        forged_bbox = Q16BBox(
            component.bbox_q16.x0 + 1,
            component.bbox_q16.y0,
            component.bbox_q16.x1,
            component.bbox_q16.y1,
        )
        components = (replace(component, bbox_q16=forged_bbox),) + components[1:]
    elif field == "mask":
        component = components[0]
        components = (replace(component, mask_digest="0" * 64),) + components[1:]
    else:
        hole = holes[0]
        holes = (replace(hole, owner_component_id="component-00000001"),)

    forged_scenario = replace(scenario, components=components, holes=holes)
    forged = replace(
        packet, scenarios=(forged_scenario,) + packet.scenarios[1:]
    )
    with pytest.raises(ValueError, match="differs from exact PNG replay"):
        verify_visual_witness_packet(forged, expected_png_bytes=original)


def test_schema_and_value_objects_reject_noncanonical_data() -> None:
    packet = extract_visual_witnesses(_panel())
    data = packet.to_data()
    data["unexpected"] = True
    with pytest.raises(ValueError, match="fields differ"):
        VisualWitnessPacket.from_data(data)

    with pytest.raises(ValueError, match="positive extent"):
        Q16BBox(1, 2, 1, 3)
    with pytest.raises(ValueError, match="canonical"):
        ComponentWitness("component-1", Q16BBox(0, 0, 1, 1), 1, "0" * 64)
    with pytest.raises(ValueError, match="canonical or null"):
        HoleWitness("hole-00000000", Q16BBox(0, 0, 1, 1), 1, "0" * 64, "x")


@pytest.mark.parametrize("bad", [b"", b"not a png", b"\x89PNG\r\n\x1a\ntruncated"])
def test_invalid_png_is_an_error_not_a_negative(bad: bytes) -> None:
    with pytest.raises(ValueError):
        extract_visual_witnesses(bad)


def test_extractor_rejects_non_bytes_transport() -> None:
    with pytest.raises(TypeError, match="exact PNG bytes"):
        extract_visual_witnesses(bytearray(_panel()))  # type: ignore[arg-type]

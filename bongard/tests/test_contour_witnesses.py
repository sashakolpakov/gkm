from __future__ import annotations

from dataclasses import replace
from io import BytesIO
import math

from PIL import Image, ImageDraw
import pytest

from bongard.contour_witnesses import (
    CONTOUR_WITNESS_CAPABILITY_IDS,
    CONTOUR_WITNESS_PACKET,
    CONTOUR_WITNESS_SCENARIO_IDS,
    ContourCountResult,
    ContourWitnessPacket,
    CountInterval,
    contour_witness_catalog_digest,
    contour_witness_extractor_digest,
    evaluate_contour_count_by_scenario,
    extract_contour_witnesses,
    verify_contour_witness_packet,
)
from bongard.legs.contracts import ValueType
from bongard.visual_witnesses import VISUAL_WITNESS_SCENARIO_IDS


def _png(
    paths: list[list[tuple[float, float]]],
    *,
    size: int = 112,
    width: int = 4,
) -> bytes:
    image = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(image)
    for path in paths:
        draw.line(path, fill="black", width=width, joint="curve")
    output = BytesIO()
    image.save(output, format="PNG", optimize=False)
    return output.getvalue()


def _topology_panel(kind: str, *, width: int = 4) -> bytes:
    if kind == "line":
        return _png([[(20, 56), (92, 56)]], width=width)
    if kind == "tee":
        return _png(
            [[(25, 28), (87, 28)], [(56, 28), (56, 88)]], width=width
        )
    if kind == "crossing":
        return _png(
            [[(22, 22), (90, 90)], [(90, 22), (22, 90)]], width=width
        )
    if kind == "loop":
        image = Image.new("RGB", (112, 112), "white")
        ImageDraw.Draw(image).ellipse((25, 25, 87, 87), outline="black", width=width)
        output = BytesIO()
        image.save(output, format="PNG", optimize=False)
        return output.getvalue()
    raise AssertionError(kind)


def _curve_panel(
    kind: str,
    *,
    size: int = 112,
    angle: float = 0.0,
    width: int = 4,
    translation: tuple[float, float] = (0.0, 0.0),
    reverse: bool = False,
) -> bytes:
    center = size / 2.0
    scale = size / 112.0
    if kind == "u":
        points = [
            (
                center + 28 * scale * math.cos(math.pi + index * math.pi / 120),
                center
                - 8 * scale
                + 28 * scale * math.sin(math.pi + index * math.pi / 120),
            )
            for index in range(121)
        ]
    elif kind == "s":
        points = []
        for index in range(121):
            parameter = -1.0 + 2.0 * index / 120
            points.append(
                (
                    center + 25 * scale * math.sin(math.pi * parameter / 2),
                    center + 32 * scale * parameter,
                )
            )
    else:
        raise AssertionError(kind)
    cosine, sine = math.cos(angle), math.sin(angle)
    dx, dy = translation
    transformed = [
        (
            center + cosine * (x - center) - sine * (y - center) + dx,
            center + sine * (x - center) + cosine * (y - center) + dy,
        )
        for x, y in points
    ]
    if reverse:
        transformed.reverse()
    return _png([transformed], size=size, width=width)


def test_packet_round_trip_and_finite_source_bound_inventory() -> None:
    packet = extract_contour_witnesses(_topology_panel("line"))

    assert CONTOUR_WITNESS_CAPABILITY_IDS == tuple(
        sorted(CONTOUR_WITNESS_CAPABILITY_IDS)
    )
    assert CONTOUR_WITNESS_SCENARIO_IDS == VISUAL_WITNESS_SCENARIO_IDS
    assert CONTOUR_WITNESS_PACKET == ValueType("contour_witness_packet")
    assert ContourWitnessPacket.from_data(packet.to_data()) == packet
    assert packet.extractor_artifact_digest == contour_witness_extractor_digest()
    assert len(contour_witness_catalog_digest()) == 64
    assert len(packet.digest()) == 64


@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        ("line", (2, 0, 0, 0)),
        ("tee", (3, 1, 0, 0)),
        ("crossing", (4, 1, 0, 1)),
        ("loop", (0, 0, 1, 0)),
    ],
)
@pytest.mark.parametrize("width", [2, 4, 6])
def test_clean_topology_is_stable_across_scenarios_and_thickness(
    kind: str, expected: tuple[int, int, int, int], width: int
) -> None:
    packet = extract_contour_witnesses(_topology_panel(kind, width=width))

    assert len(packet.scenarios) == 3
    for scenario in packet.scenarios:
        assert len(scenario.contours) == 1
        contour = scenario.contours[0]
        observed = (
            contour.endpoint_count.lower,
            contour.branchpoint_count.lower,
            contour.cycle_count.lower,
            contour.crossing_count.lower,
        )
        assert observed == expected
        assert contour.endpoint_count.exact
        assert contour.branchpoint_count.exact
        assert contour.cycle_count.exact
        assert contour.crossing_count.exact
        assert contour.topology_disposition == "determinate"
        assert contour.contour_id == "contour-00000000"
        assert contour.owner_component_id == "component-00000000"


def test_crossing_is_an_x_junction_not_every_branchpoint() -> None:
    tee = extract_contour_witnesses(_topology_panel("tee"))
    crossing = extract_contour_witnesses(_topology_panel("crossing"))

    for scenario in tee.scenarios:
        contour = scenario.contours[0]
        assert contour.branchpoints[0].incident_arm_count == CountInterval.point(3)
        assert contour.crossing_branchpoint_ids == ()
    for scenario in crossing.scenarios:
        contour = scenario.contours[0]
        assert contour.branchpoints[0].incident_arm_count == CountInterval.point(4)
        assert contour.crossing_branchpoint_ids == ("branchpoint-00000000",)


def test_five_arm_raster_is_uncertain_not_a_certified_non_crossing() -> None:
    center = (56.0, 56.0)
    paths = []
    for degrees in (0, 72, 144, 216, 288):
        angle = math.radians(degrees)
        paths.append(
            [center, (56 + 38 * math.cos(angle), 56 + 38 * math.sin(angle))]
        )
    packet = extract_contour_witnesses(_png(paths, width=2))
    result = evaluate_contour_count_by_scenario(
        packet, "topology.crossing_count", 0
    )

    assert all(item.observed == CountInterval(0, 1) for item in result.observations)
    assert all(item.disposition == "indeterminate" for item in result.observations)


@pytest.mark.parametrize(
    ("kind", "size", "angle", "width", "translation"),
    [
        ("u", 80, 0.25, 2, (3.0, -2.0)),
        ("u", 112, 1.5, 4, (-4.0, 3.0)),
        ("u", 160, 0.4, 8, (5.0, 4.0)),
        ("s", 80, 0.25, 2, (3.0, -2.0)),
        ("s", 112, 1.5, 4, (-4.0, 3.0)),
        ("s", 160, 0.4, 8, (5.0, 4.0)),
    ],
)
def test_s_and_u_curvature_survive_rerender_rotation_translation_and_thickness(
    kind: str,
    size: int,
    angle: float,
    width: int,
    translation: tuple[float, float],
) -> None:
    packet = extract_contour_witnesses(
        _curve_panel(
            kind,
            size=size,
            angle=angle,
            width=width,
            translation=translation,
        )
    )

    for scenario in packet.scenarios:
        curvature = scenario.contours[0].curvature
        assert curvature.curve_class == f"{kind}-like"
        if kind == "s":
            assert curvature.reversal_count.lower >= 1
            assert curvature.run_count.lower >= 2
        else:
            assert curvature.reversal_count == CountInterval.point(0)
            assert curvature.run_count == CountInterval.point(1)


@pytest.mark.parametrize("kind", ["s", "u"])
def test_curvature_counts_are_invariant_to_path_traversal(kind: str) -> None:
    forward = extract_contour_witnesses(
        _curve_panel(kind, angle=0.4, width=4, reverse=False)
    )
    reverse = extract_contour_witnesses(
        _curve_panel(kind, angle=0.4, width=4, reverse=True)
    )

    for first, second in zip(forward.scenarios, reverse.scenarios, strict=True):
        left = first.contours[0].curvature
        right = second.contours[0].curvature
        assert left.reversal_count == right.reversal_count
        assert left.run_count == right.run_count
        assert left.curve_class == right.curve_class


def test_count_evaluation_retains_scenarios_and_interval_semantics() -> None:
    packet = extract_contour_witnesses(_topology_panel("crossing"))
    present = evaluate_contour_count_by_scenario(
        packet, "topology.crossing_count", 1
    )
    absent = evaluate_contour_count_by_scenario(
        packet, "topology.crossing_count", 2
    )

    assert ContourCountResult.from_data(present.to_data()) == present
    assert [item.disposition for item in present.observations] == ["present"] * 3
    assert [item.disposition for item in absent.observations] == [
        "certified_absent"
    ] * 3


def test_exact_replay_rejects_mask_and_panel_digest_forgery() -> None:
    original = _topology_panel("crossing")
    packet = extract_contour_witnesses(original)
    scenario = packet.scenarios[0]
    contour = scenario.contours[0]
    forged_contour = replace(contour, skeleton_digest="0" * 64)
    forged_scenario = replace(
        scenario, contours=(forged_contour,) + scenario.contours[1:]
    )
    forged_nested = replace(
        packet, scenarios=(forged_scenario,) + packet.scenarios[1:]
    )
    forged_panel = replace(packet, panel_digest="0" * 64)

    assert verify_contour_witness_packet(packet, expected_png_bytes=original) is packet
    with pytest.raises(ValueError, match="differs from exact PNG replay"):
        verify_contour_witness_packet(forged_nested, expected_png_bytes=original)
    with pytest.raises(ValueError, match="differs from exact PNG replay"):
        verify_contour_witness_packet(forged_panel, expected_png_bytes=original)


def test_replay_rejects_different_exact_png_bytes() -> None:
    original = _topology_panel("crossing")
    packet = extract_contour_witnesses(original)
    with pytest.raises(ValueError, match="differs from exact PNG replay"):
        verify_contour_witness_packet(
            packet, expected_png_bytes=_topology_panel("tee")
        )


@pytest.mark.parametrize("bad", [b"", b"not png", b"\x89PNG\r\n\x1a\ntruncated"])
def test_invalid_png_and_nonbyte_transport_are_errors(bad: bytes) -> None:
    with pytest.raises(ValueError):
        extract_contour_witnesses(bad)
    with pytest.raises(TypeError, match="exact PNG bytes"):
        extract_contour_witnesses(bytearray(_topology_panel("line")))  # type: ignore[arg-type]


def test_strict_dtos_reject_extra_fields_and_invalid_intervals() -> None:
    packet = extract_contour_witnesses(_topology_panel("line"))
    data = packet.to_data()
    data["unexpected"] = True
    with pytest.raises(ValueError, match="fields differ"):
        ContourWitnessPacket.from_data(data)
    with pytest.raises(ValueError, match="lower exceeds upper"):
        CountInterval(2, 1)

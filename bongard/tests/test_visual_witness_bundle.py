from __future__ import annotations

from dataclasses import replace
from io import BytesIO

from PIL import Image, ImageDraw
import pytest

from bongard.legs.contracts import ValueType
from bongard.visual_witness_bundle import (
    VISUAL_WITNESS_BUNDLE,
    VISUAL_WITNESS_BUNDLE_EXTRACTOR_ID,
    VISUAL_WITNESS_BUNDLE_VERSION,
    VisualWitnessBundle,
    extract_visual_witness_bundle,
    verify_visual_witness_bundle,
    visual_witness_bundle_catalog_digest,
    visual_witness_bundle_extractor_digest,
)


def _panel(*, second_component: bool = True) -> bytes:
    image = Image.new("RGB", (64, 64), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((6, 8, 30, 44), fill="black")
    draw.rectangle((12, 14, 24, 38), fill="white")
    if second_component:
        draw.rectangle((44, 20, 54, 34), fill="black")
    output = BytesIO()
    image.save(output, format="PNG", optimize=False)
    return output.getvalue()


def test_bundle_round_trip_identity_and_child_alignment() -> None:
    bundle = extract_visual_witness_bundle(_panel())

    assert VISUAL_WITNESS_BUNDLE == ValueType("visual_witness_bundle")
    assert VISUAL_WITNESS_BUNDLE_EXTRACTOR_ID == "visual-witness-bundle"
    assert VISUAL_WITNESS_BUNDLE_VERSION == "1"
    assert VisualWitnessBundle.from_data(bundle.to_data()) == bundle
    assert bundle.base_packet.panel_digest == bundle.contour_packet.panel_digest
    assert bundle.panel_digest == bundle.base_packet.panel_digest
    assert bundle.width_pixels == bundle.base_packet.width_pixels
    assert bundle.height_pixels == bundle.base_packet.height_pixels
    assert bundle.assembler_artifact_digest == visual_witness_bundle_extractor_digest()
    assert len(visual_witness_bundle_catalog_digest()) == 64
    for base, contour in zip(
        bundle.base_packet.scenarios, bundle.contour_packet.scenarios, strict=True
    ):
        assert base.scenario_id == contour.scenario_id
        assert tuple(item.component_id for item in base.components) == tuple(
            item.owner_component_id for item in contour.contours
        )


def test_bundle_rejects_child_packets_from_different_exact_panels() -> None:
    first = extract_visual_witness_bundle(_panel(second_component=True))
    second = extract_visual_witness_bundle(_panel(second_component=False))

    with pytest.raises(ValueError, match="same exact PNG"):
        replace(first, contour_packet=second.contour_packet)


def test_exact_replay_rejects_nested_forgery_and_other_bytes() -> None:
    panel = _panel()
    bundle = extract_visual_witness_bundle(panel)
    scenario = bundle.contour_packet.scenarios[0]
    contour = scenario.contours[0]
    forged_contour = replace(contour, skeleton_digest="0" * 64)
    forged_scenario = replace(
        scenario, contours=(forged_contour,) + scenario.contours[1:]
    )
    forged_contour_packet = replace(
        bundle.contour_packet,
        scenarios=(forged_scenario,) + bundle.contour_packet.scenarios[1:],
    )
    forged_bundle = replace(bundle, contour_packet=forged_contour_packet)

    assert verify_visual_witness_bundle(bundle, panel) is bundle
    with pytest.raises(ValueError, match="differs from exact PNG replay"):
        verify_visual_witness_bundle(forged_bundle, panel)
    with pytest.raises(ValueError, match="differs from exact PNG replay"):
        verify_visual_witness_bundle(
            bundle, _panel(second_component=False)
        )


def test_bundle_schema_and_transport_fail_closed() -> None:
    bundle = extract_visual_witness_bundle(_panel())
    data = bundle.to_data()
    data["unexpected"] = True
    with pytest.raises(ValueError, match="fields differ"):
        VisualWitnessBundle.from_data(data)
    with pytest.raises(TypeError, match="exact PNG bytes"):
        extract_visual_witness_bundle(bytearray(_panel()))  # type: ignore[arg-type]

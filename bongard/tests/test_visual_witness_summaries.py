from __future__ import annotations

from dataclasses import replace
from io import BytesIO

from PIL import Image, ImageDraw
import pytest

from bongard.blind_soft_transport import canonical_witness_summaries
import bongard.visual_witness_summaries as summaries_module
from bongard.visual_witness_summaries import (
    visual_joint_soft_witness_interface_digest,
    visual_soft_witness_interface_digest,
    visual_witness_summaries,
    visual_witness_summary_artifact_digest,
)
from bongard.visual_witness_bundle import extract_visual_witness_bundle
from bongard.visual_witnesses import Q16BBox, extract_visual_witnesses


def _panel() -> bytes:
    image = Image.new("RGB", (64, 48), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((5, 6, 30, 40), fill="black")
    draw.rectangle((11, 12, 24, 34), fill="white")
    draw.rectangle((42, 18, 54, 31), fill="black")
    output = BytesIO()
    image.save(output, format="PNG", optimize=False)
    return output.getvalue()


def test_summaries_are_deterministic_complete_neutral_and_globally_unique() -> None:
    panel = _panel()
    first = visual_witness_summaries(
        extract_visual_witness_bundle(panel),
        expected_png_bytes=panel,
    )
    second = visual_witness_summaries(
        extract_visual_witness_bundle(panel),
        expected_png_bytes=panel,
    )

    assert first == second == tuple(sorted(first))
    # geometry + 3 * (base counts + topology counts + 2 contours + 1 hole)
    assert len(first) == 16
    ids = tuple(witness_id for witness_id, _ in first)
    assert len(ids) == len(set(ids))
    assert ids.count("panel:geometry") == 1
    assert sum(witness_id.endswith(":counts") for witness_id in ids) == 3
    assert sum(witness_id.endswith(":topology-counts") for witness_id in ids) == 3
    assert sum(":contour:" in witness_id for witness_id in ids) == 6
    assert sum(":hole:" in witness_id for witness_id in ids) == 3

    rendered = "\n".join(f"{key} {value}" for key, value in first).lower()
    assert "width_pixels=64" in rendered
    assert "height_pixels=48" in rendered
    assert "component_count=2" in rendered
    assert "owned_hole_count=1" in rendered
    assert "bbox_q16" in rendered
    assert "owner_component_id=component-" in rendered
    assert "endpoint_count=" in rendered
    assert "signed_curvature_reversal_count=" in rendered
    assert "curve_class=" in rendered
    for forbidden in (
        "task_id",
        "support side",
        "positive panel",
        "negative panel",
        "query role",
        "source path",
    ):
        assert forbidden not in rendered


def test_output_is_accepted_without_reordering_by_blind_transport() -> None:
    summaries = visual_witness_summaries(extract_visual_witness_bundle(_panel()))

    accepted = canonical_witness_summaries(summaries)
    assert tuple((item.witness_id, item.description) for item in accepted) == summaries


def test_source_and_composite_digests_are_sensitive_to_every_bound_layer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summary_digest = visual_witness_summary_artifact_digest()
    interface_digest = visual_soft_witness_interface_digest()
    assert interface_digest == visual_joint_soft_witness_interface_digest()
    assert len(summary_digest) == len(interface_digest) == 64

    monkeypatch.setattr(summaries_module, "_source_digest", lambda: "0" * 64)
    changed_summary = visual_witness_summary_artifact_digest()
    assert changed_summary != summary_digest
    monkeypatch.undo()

    dependencies = (
        "visual_witness_bundle_extractor_digest",
        "visual_witness_bundle_catalog_digest",
        "visual_witness_summary_artifact_digest",
    )
    for dependency in dependencies:
        original = getattr(summaries_module, dependency)
        monkeypatch.setattr(summaries_module, dependency, lambda: "f" * 64)
        assert visual_soft_witness_interface_digest() != interface_digest
        monkeypatch.setattr(summaries_module, dependency, original)


def test_exact_replay_rejects_a_geometry_tampered_packet() -> None:
    panel = _panel()
    bundle = extract_visual_witness_bundle(panel)
    packet = bundle.base_packet
    scenario = packet.scenarios[0]
    component = scenario.components[0]
    bbox = component.bbox_q16
    forged_component = replace(
        component,
        bbox_q16=Q16BBox(bbox.x0 + 1, bbox.y0, bbox.x1, bbox.y1),
    )
    forged_scenario = replace(
        scenario,
        components=(forged_component, *scenario.components[1:]),
    )
    forged_packet = replace(
        packet,
        scenarios=(forged_scenario, *packet.scenarios[1:]),
    )
    forged_bundle = replace(bundle, base_packet=forged_packet)

    with pytest.raises(ValueError, match="differs from exact PNG replay"):
        visual_witness_summaries(forged_bundle, expected_png_bytes=panel)


def test_summary_boundary_requires_the_joint_bundle() -> None:
    with pytest.raises(TypeError, match="VisualWitnessBundle"):
        visual_witness_summaries(extract_visual_witnesses(_panel()))  # type: ignore[arg-type]


def test_large_detail_inventory_is_deterministically_bounded_to_512() -> None:
    image = Image.new("RGB", (128, 128), "white")
    draw = ImageDraw.Draw(image)
    for y in range(4, 124, 8):
        for x in range(4, 124, 8):
            draw.point((x, y), fill="black")
    output = BytesIO()
    image.save(output, format="PNG", optimize=False)
    panel = output.getvalue()

    summaries = visual_witness_summaries(extract_visual_witness_bundle(panel))

    assert len(summaries) == 512
    assert any(key == "panel:inventory-bounds" for key, _ in summaries)
    assert all(len(description.encode("utf-8")) <= 512 for _, description in summaries)

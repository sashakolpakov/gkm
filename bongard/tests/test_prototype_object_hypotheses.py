from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image, ImageDraw
import pytest

import bongard.prototype_object_hypotheses as module
from bongard.prototype_object_hypotheses import (
    ATLAS_MAX_SHEETS,
    ATLAS_SLOT_CAPACITY,
    ObjectHypothesisError,
    ObjectHypothesisPacket,
    extract_object_hypothesis_packet,
    render_object_hypothesis_atlas,
    verify_object_hypothesis_packet,
)


def _png(image: Image.Image) -> bytes:
    output = BytesIO()
    image.save(output, format="PNG", optimize=False)
    return output.getvalue()


def _live_like_panel() -> bytes:
    """Produce exact scenario counts 14/4/17 with n-1 linkage clusters."""

    image = Image.new("RGB", (360, 128), "white")
    draw = ImageDraw.Draw(image)

    # Seventeen black fragments.  Strength-80 bridges make four components in
    # the middle scenario but disappear in the high-threshold scenario.
    groups = (
        (10, 18, 26, 34),
        (50, 58, 66, 74),
        (90, 98, 106, 114, 122),
        (170, 178, 186, 270),
    )
    for starts in groups:
        draw.rectangle((starts[0], 21, starts[-1] + 3, 22), fill=(175, 175, 175))
        for x0 in starts:
            draw.rectangle((x0, 20, x0 + 3, 23), fill="black")

    # Ten strength-40 components exist only in the low-threshold scenario.
    # Nine form a nearer chain; the tenth is the final outlier.
    for x0 in (10, 40, 70, 100, 130, 160, 190, 220, 250, 340):
        draw.rectangle((x0, 80, x0 + 3, 83), fill=(215, 215, 215))
    return _png(image)


def _blank_panel() -> bytes:
    return _png(Image.new("RGB", (32, 32), "white"))


def test_live_like_multicomponent_catalog_recovers_n_minus_one_candidates() -> None:
    packet = extract_object_hypothesis_packet(_live_like_panel())

    singleton_counts = tuple(
        sum(len(item.source_component_ids) == 1 for item in scenario.hypotheses)
        for scenario in packet.scenarios
    )
    assert singleton_counts == (14, 4, 17)
    for scenario, expected_cluster_size in zip(
        packet.scenarios, (13, 3, 16), strict=True
    ):
        candidates = [
            item
            for item in scenario.hypotheses
            if len(item.source_component_ids) == expected_cluster_size
        ]
        assert len(candidates) == 1
        candidate = candidates[0]
        assert candidate.emergence_gap_pixels > 0
        singleton_area = {
            item.source_component_ids[0]: item.union_area_pixels
            for item in scenario.hypotheses
            if len(item.source_component_ids) == 1
        }
        assert candidate.union_area_pixels == sum(
            singleton_area[item] for item in candidate.source_component_ids
        )

    assert packet.to_data()["candidate_policy"] == {
        "connected_components_are_low_level_regions": True,
        "hypotheses_are_candidate_groupings": True,
        "candidate_independent_of_profile_and_rubric": True,
        "semantic_object_completeness_claimed": False,
        "omission_on_atlas_overflow": False,
    }
    assert packet.to_data()["runtime_authority"]["lean_present"] is False


def test_packet_round_trip_atlas_is_font_free_opaque_and_exactly_replayable() -> None:
    panel = _live_like_panel()
    packet = extract_object_hypothesis_packet(panel)

    assert ObjectHypothesisPacket.from_data(packet.to_data()) == packet
    first = render_object_hypothesis_atlas(packet, panel)
    second = render_object_hypothesis_atlas(packet, panel)
    assert first == second
    assert tuple(name for name, _ in first) == tuple(
        f"sheet_{index:03d}.png" for index in range(len(first))
    )
    assert all(data.startswith(b"\x89PNG\r\n\x1a\n") for _, data in first)
    assert tuple(len(sheet.slots) for sheet in packet.atlas_sheets[:-1]) == (
        ATLAS_SLOT_CAPACITY,
    ) * max(0, len(packet.atlas_sheets) - 1)
    assert sum(len(sheet.slots) for sheet in packet.atlas_sheets) == sum(
        len(scenario.hypotheses) for scenario in packet.scenarios
    )
    assert verify_object_hypothesis_packet(
        packet,
        panel,
        expected_atlas_png_by_name=dict(first),
    ) is packet
    for _, data in first:
        with Image.open(BytesIO(data)) as image:
            assert image.mode == "L"
            assert image.size == (module.ATLAS_WIDTH_PIXELS, module.ATLAS_HEIGHT_PIXELS)
            image.verify()


def test_blank_panel_has_one_canonical_empty_sheet() -> None:
    panel = _blank_panel()
    first = extract_object_hypothesis_packet(panel)
    second = extract_object_hypothesis_packet(panel)

    assert all(not scenario.hypotheses for scenario in first.scenarios)
    assert len(first.atlas_sheets) == 1
    assert first.atlas_sheets[0].name == "sheet_000.png"
    assert first.atlas_sheets[0].slots == ()
    assert first == second
    assert render_object_hypothesis_atlas(first, panel) == (
        render_object_hypothesis_atlas(second, panel)
    )


def test_tampering_and_other_pixel_replay_are_errors_not_absence() -> None:
    panel = _live_like_panel()
    packet = extract_object_hypothesis_packet(panel)
    data = packet.to_data()
    data["scenarios"][0]["hypotheses"][0]["masked_crop_pixel_digest"] = "0" * 64
    with pytest.raises(ObjectHypothesisError):
        ObjectHypothesisPacket.from_data(data)

    atlas = dict(render_object_hypothesis_atlas(packet, panel))
    name = next(iter(atlas))
    atlas[name] = atlas[name][:-1] + bytes((atlas[name][-1] ^ 1,))
    with pytest.raises(ObjectHypothesisError, match="atlas bytes differ"):
        verify_object_hypothesis_packet(
            packet, panel, expected_atlas_png_by_name=atlas
        )
    with pytest.raises(ObjectHypothesisError, match="differs from PNG replay"):
        verify_object_hypothesis_packet(packet, _blank_panel())


def test_32_sheet_guard_fails_closed_instead_of_omitting_candidates() -> None:
    capacity = ATLAS_MAX_SHEETS * ATLAS_SLOT_CAPACITY
    module._require_atlas_capacity(capacity)
    with pytest.raises(
        ObjectHypothesisError, match="exceeds 32 sheets; omission is forbidden"
    ):
        module._require_atlas_capacity(capacity + 1)


def test_non_bytes_transport_is_rejected() -> None:
    with pytest.raises(TypeError, match="exact PNG bytes"):
        extract_object_hypothesis_packet(bytearray(_blank_panel()))  # type: ignore[arg-type]


from __future__ import annotations

from copy import deepcopy
import hashlib
from inspect import signature
from io import BytesIO

import numpy as np
from PIL import Image
import pytest

import bongard.composite_visual_packet as composite
from bongard.composite_visual_packet import (
    BilateralSymmetryWitnessPacket,
    ExactPanelWitnessPacket,
    extract_exact_panel_witness_packet,
    verify_exact_panel_witness_packet,
)
from bongard.evidence import Disposition


def _png(panel: np.ndarray) -> bytes:
    encoded = BytesIO()
    Image.fromarray(panel, mode="L").save(encoded, format="PNG")
    return encoded.getvalue()


def _two_square_loops() -> bytes:
    panel = np.full((96, 96), 255, dtype=np.uint8)
    panel[10:35, 10:35] = 0
    panel[15:30, 15:30] = 255
    panel[45:88, 45:88] = 0
    panel[50:83, 50:83] = 255
    return _png(panel)


def test_exact_panel_extractor_is_candidate_independent_and_coherent() -> None:
    assert tuple(signature(extract_exact_panel_witness_packet).parameters) == (
        "png_bytes",
    )
    raw = _two_square_loops()
    first = extract_exact_panel_witness_packet(raw)
    second = extract_exact_panel_witness_packet(raw)

    assert first == second
    assert first.panel_digest == hashlib.sha256(raw).hexdigest()
    assert first.visual_bundle.panel_digest == first.loop_scene.panel_digest
    assert first.loop_scene.panel_digest == first.bilateral_symmetry.panel_digest
    assert first.loop_scene.parent_bundle_digest == first.visual_bundle.digest()
    assert (
        first.bilateral_symmetry.parent_visual_bundle_digest
        == first.visual_bundle.digest()
    )
    assert ExactPanelWitnessPacket.from_data(first.to_data()) == first
    assert verify_exact_panel_witness_packet(first, expected_png_bytes=raw) == first

    changed = bytearray(raw)
    # A distinct valid PNG encoding of the same raster still has a distinct
    # exact-byte identity.  Pillow's optimize flag changes the container bytes.
    panel = np.asarray(Image.open(BytesIO(raw)).convert("L"), dtype=np.uint8)
    encoded = BytesIO()
    Image.fromarray(panel, mode="L").save(encoded, format="PNG", optimize=True)
    changed_raw = encoded.getvalue()
    assert changed_raw != bytes(changed)
    assert extract_exact_panel_witness_packet(changed_raw).panel_digest != (
        first.panel_digest
    )


def test_strict_child_serialization_rejects_digest_and_interval_tampering() -> None:
    packet = extract_exact_panel_witness_packet(_two_square_loops())

    wrong_panel = deepcopy(packet.to_data())
    wrong_panel["loop_scene"]["panel_digest"] = "0" * 64
    with pytest.raises(ValueError, match="same exact PNG|parent|different"):
        ExactPanelWitnessPacket.from_data(wrong_panel)

    wrong_interval = deepcopy(packet.to_data())
    scenario = wrong_interval["bilateral_symmetry"]["scenarios"][0]
    assert scenario["coverage_ppm"] is not None
    scenario["coverage_ppm"]["lower"] -= 1
    with pytest.raises(ValueError, match="provenance"):
        ExactPanelWitnessPacket.from_data(wrong_interval)

    extra_field = deepcopy(packet.bilateral_symmetry.to_data())
    extra_field["polarity"] = "negative"
    with pytest.raises(ValueError, match="fields differ"):
        BilateralSymmetryWitnessPacket.from_data(extra_field)


def test_failed_bilateral_measurement_is_error_never_absence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken_measurement(_: np.ndarray) -> object:
        raise RuntimeError("synthetic extractor failure")

    monkeypatch.setattr(composite, "_score_mask", broken_measurement)
    packet = extract_exact_panel_witness_packet(_two_square_loops())

    assert tuple(
        item.disposition for item in packet.bilateral_symmetry.scenarios
    ) == (Disposition.ERROR,) * 3
    assert all(
        item.error_type == "RuntimeError"
        and "synthetic extractor failure" in (item.reason or "")
        for item in packet.bilateral_symmetry.scenarios
    )
    assert all(
        item.disposition is not Disposition.CERTIFIED_ABSENT
        for item in packet.bilateral_symmetry.scenarios
    )


def test_blank_and_unmeasurable_panels_keep_distinct_dispositions() -> None:
    blank = extract_exact_panel_witness_packet(
        _png(np.full((64, 64), 255, dtype=np.uint8))
    )
    tiny_panel = np.full((64, 64), 255, dtype=np.uint8)
    tiny_panel[30:34, 30:34] = 0
    tiny = extract_exact_panel_witness_packet(_png(tiny_panel))

    assert tuple(
        item.disposition for item in blank.bilateral_symmetry.scenarios
    ) == (Disposition.CERTIFIED_ABSENT,) * 3
    assert tuple(
        item.disposition for item in tiny.bilateral_symmetry.scenarios
    ) == (Disposition.INDETERMINATE,) * 3
    assert all(
        item.coverage_ppm == composite.PpmInterval(0, composite.PPM_SCALE)
        for item in tiny.bilateral_symmetry.scenarios
    )

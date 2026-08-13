from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import pytest

from bongard import panel_action_count_synthetic_identifiability as fixtures
from bongard import panel_action_count_synthetic_pooled_control as subject


EXPECTED_SOURCE_SHA256 = "1d2ea4a097e98ba4b4d69de316b20d0fb1cb8619517d0254ce7272dbeb84d021"
EXPECTED_ASYMMETRIC_PNG_SHA256 = (
    "c3e497475d29c6e0166f154a1f7041a8181c1707d05da41c5a898562c50cd19f"
)
EXPECTED_ASYMMETRIC_VECTOR_SHA256 = (
    "87bf4c715be8c138a862661969e4fbb7f83c916af07f149201b96f0e9db12fd7"
)


def _asymmetric_program() -> fixtures.Program:
    return fixtures.Program(
        "pooled-control-test",
        (
            fixtures.LineAction(
                "line-0", fixtures.Point(96, 256), fixtures.Point(576, 256)
            ),
            fixtures.ArcAction(
                "arc-0",
                fixtures.Point(576, 256),
                fixtures.Point(800, 400),
                fixtures.Point(736, 768),
            ),
        ),
    )


def _arbitrary_png() -> bytes:
    image = Image.new("L", (32, 32), 255)
    ImageDraw.Draw(image).line((4, 16, 27, 16), fill=0, width=2)
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def test_exact_feature_vocabulary_and_source_boundary() -> None:
    assert subject.source_sha256() == EXPECTED_SOURCE_SHA256
    assert len(subject.SCALE_SPECS) == 4
    assert len(subject.PER_SCALE_FEATURE_NAMES) == 28
    assert len(subject.FEATURE_NAMES) == len(set(subject.FEATURE_NAMES)) == 112
    assert subject.FEATURE_NAMES == tuple(
        f"{scale[0]}:{name}"
        for scale in subject.SCALE_SPECS
        for name in subject.PER_SCALE_FEATURE_NAMES
    )
    addresses = subject.dependency_source_addresses()
    assert set(addresses) == {
        "bongard.panel_action_count_synthetic_pooled_control",
        "bongard.panel_action_count_synthetic_identifiability",
        "bongard.runtime_source_snapshot",
    }
    assert addresses == {
        "bongard.panel_action_count_synthetic_pooled_control": (
            "sha256:" + EXPECTED_SOURCE_SHA256
        ),
        "bongard.panel_action_count_synthetic_identifiability": (
            "sha256:7e52729cfda9effd831352802c47f3cf7f8d29f00da3a90dc7250bf1cdc722bf"
        ),
        "bongard.runtime_source_snapshot": (
            "sha256:67d37b28497e589f6766367a73a71bb3f6fe70510436123d5dac7730fc681ced"
        ),
    }
    assert set(subject.runtime_fingerprint()) == {
        "byteorder", "machine", "python", "platform", "pillow", "numpy",
        "scipy", "scikit_learn",
    }
    source = Path(subject.__file__).read_text(encoding="utf-8")
    forbidden = "panel_action_count_" + "skeleton_graph_dev_command"
    assert forbidden not in source
    assert not hasattr(subject, "main")


def test_issued_panel_matches_frozen_reference_feature_bytes() -> None:
    # This digest was frozen when the neutral extractor was independently
    # checked byte-for-byte against the historical implementation.  The
    # terminal historical module is deliberately not imported or executed by
    # this synthetic-only regression.
    panel = fixtures.render_program(
        _asymmetric_program(), fixtures.Nuisance("mirror_x_r90", 3, 1000)
    )
    actual = subject.extract_feature_vector(panel.png_bytes)
    assert hashlib.sha256(panel.png_bytes).hexdigest() == EXPECTED_ASYMMETRIC_PNG_SHA256
    assert actual.shape == (112,)
    assert actual.dtype == np.float32
    assert actual.flags.c_contiguous
    assert hashlib.sha256(actual.tobytes(order="C")).hexdigest() == (
        EXPECTED_ASYMMETRIC_VECTOR_SHA256
    )
    assert np.count_nonzero(actual) == 92


def test_arbitrary_png_and_nonexact_payload_types_are_rejected() -> None:
    with pytest.raises(subject.SyntheticPooledControlError, match="not issued"):
        subject.extract_feature_vector(_arbitrary_png())
    for malformed in (bytearray(_arbitrary_png()), memoryview(_arbitrary_png()), "png"):
        with pytest.raises(subject.SyntheticPooledControlError, match="exact bytes"):
            subject.extract_feature_vector(malformed)  # type: ignore[arg-type]


def test_explicit_synthetic_issuer_seam_is_vector_exact_and_label_free() -> None:
    panel = fixtures.render_program(
        fixtures._carrier_program(fixtures.CountPair(2, 1), "radial")
    )
    direct = subject.extract_feature_vector(panel.png_bytes)
    seen: list[bytes] = []

    def issuer(raw: bytes) -> str:
        seen.append(raw)
        return fixtures.require_issued_synthetic_png(raw)

    bridged = subject.extract_issued_feature_vector(
        panel.png_bytes, require_issued=issuer
    )
    assert seen == [panel.png_bytes]
    assert bridged.tobytes() == direct.tobytes()

    with pytest.raises(subject.SyntheticPooledControlError, match="wrong PNG"):
        subject.extract_issued_feature_vector(
            panel.png_bytes, require_issued=lambda _raw: "sha256:" + "0" * 64
        )
    with pytest.raises(subject.SyntheticPooledControlError, match="input differs"):
        subject.extract_issued_feature_vector(
            panel.png_bytes, require_issued=object()
        )


def test_all_d4_issued_views_produce_finite_nontrivial_vectors() -> None:
    vectors = []
    for d4 in fixtures.D4_NAMES:
        panel = fixtures.render_program(
            _asymmetric_program(), fixtures.Nuisance(d4, 2, 1000)
        )
        vector = subject.extract_feature_vector(panel.png_bytes)
        assert vector.shape == (112,)
        assert vector.dtype == np.float32
        assert np.isfinite(vector).all()
        assert np.count_nonzero(vector) >= 40
        vectors.append(vector)
    assert len({vector.tobytes(order="C") for vector in vectors}) >= 4

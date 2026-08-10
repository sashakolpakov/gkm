from __future__ import annotations

from io import BytesIO
from pathlib import Path
from types import MappingProxyType

import numpy as np
from PIL import Image, ImageDraw
import pytest

from bongard import panel_action_count_skeleton_graph_inference_custody as subject


SHA_A = "sha256:" + "a" * 64
SHA_B = "sha256:" + "b" * 64


def _png(kind: str) -> bytes:
    image = Image.new("L", (32, 32), 255)
    draw = ImageDraw.Draw(image)
    if kind == "line":
        draw.line((4, 7, 27, 24), fill=0, width=2)
    else:
        draw.ellipse((5, 5, 26, 26), outline=0, width=2)
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


class _FakeModel:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def predict(self, *, head: str, features: np.ndarray) -> np.ndarray:
        self.calls.append(head)
        size = (
            len(subject.DIRECT_PAIR_CLASS_ORDER)
            if head == "direct_pair"
            else len(subject.CATALOG_CLASS_ORDER)
        )
        base = np.arange(1, size + 1, dtype=np.float64)[None, :]
        modulation = np.abs(features[:, :1].astype(np.float64)) + 1.0
        result = base * modulation
        return result / result.sum(axis=1, keepdims=True)


def _bindings() -> MappingProxyType:
    return MappingProxyType(
        {
            "core_source_sha256": subject.core.source_sha256(),
            "core_config_digest": subject.core.config_digest(),
            "model_file_sha256": SHA_A,
            "passed_fit_protocol_record_digest": SHA_B,
            "passed_fit_authority_source_sha256": (
                "sha256:" + subject.passed_fit_module.source_sha256()
            ),
            "passed_fit_algorithm_digest": (
                subject.passed_fit_module.PASSED_FIT_ALGORITHM_DIGEST
            ),
            "inference_source_sha256": subject.source_sha256(),
            "inference_algorithm_digest": subject.algorithm_digest(),
        }
    )


def _install_fake_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> _FakeModel:
    model = _FakeModel()
    authority = subject._AuthenticatedInferenceAuthority(
        passed_fit=None,  # type: ignore[arg-type]
        model=model,  # type: ignore[arg-type]
        bindings=_bindings(),
    )
    monkeypatch.setattr(
        subject, "_authenticate_inference_authority", lambda *args, **kwargs: authority
    )
    return model


def _creator_kwargs() -> dict[str, object]:
    return {
        "passed_fit": object(),
        "development_precommit_path": Path("unused-precommit"),
        "development_result_path": Path("unused-result"),
        "development_replay_path": Path("unused-replay"),
        "model_path": Path("unused-model"),
        "feature_artifact_path": Path("unused-features"),
        "prediction_artifact_path": Path("unused-predictions"),
    }


def _archive_addresses(value) -> tuple[bytes, str, str]:
    raw = value.to_bytes()
    return raw, subject._raw_address(raw), value.record_digest


def test_role_free_batch_deduplicates_and_cold_replays_without_inference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _install_fake_authority(monkeypatch)
    line, circle = _png("line"), _png("circle")
    kwargs = _creator_kwargs()
    batch = subject.create_raw_inference_batch(
        **kwargs, png_payloads=(circle, line, line)
    )
    assert batch.unique_png_count == 2
    assert batch.input_occurrence_count == 3
    assert tuple(row.png_sha256 for row in batch.rows) == tuple(
        sorted((subject._raw_address(line), subject._raw_address(circle)))
    )
    by_digest = {row.png_sha256: row for row in batch.rows}
    assert by_digest[subject._raw_address(line)].occurrence_count == 2
    assert by_digest[subject._raw_address(circle)].occurrence_count == 1
    assert all(len(row.feature_values) == 112 for row in batch.rows)
    assert all(len(row.direct_pair_probabilities) == 33 for row in batch.rows)
    assert all(len(row.catalog_probabilities) == 3 for row in batch.rows)
    assert model.calls == ["direct_pair", "catalog_three_class"]
    assert subject.SkeletonGraphRawInferenceBatch.from_data(batch.to_data()) == batch

    banned = {"task", "path", "side", "ordinal", "role", "formula", "label", "panel"}

    def check_keys(value: object) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                assert not (set(key.lower().split("_")) & banned), key
                check_keys(item)
        elif isinstance(value, list):
            for item in value:
                check_keys(item)

    check_keys(batch.to_data())

    receipt = subject.fresh_verify_raw_inference_batch(
        batch, **kwargs, png_payloads=(line, circle, line)
    )
    assert receipt.exact_recompute is True
    assert receipt.feature_extraction_calls == 2
    assert receipt.model_prediction_api_calls == 2
    assert receipt.estimator_predict_proba_calls == 4
    assert model.calls == [
        "direct_pair", "catalog_three_class", "direct_pair", "catalog_three_class"
    ]

    batch_raw, batch_file, batch_record = _archive_addresses(batch)
    receipt_raw, receipt_file, receipt_record = _archive_addresses(receipt)
    monkeypatch.setattr(
        subject.core,
        "extract_feature_vector",
        lambda _raw: (_ for _ in ()).throw(AssertionError("pixel decode during cold replay")),
    )
    monkeypatch.setattr(
        subject,
        "_authenticate_inference_authority",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("model authentication during cold replay")
        ),
    )
    replay = subject.cold_replay_raw_inference(
        raw_batch_bytes=batch_raw,
        recompute_receipt_bytes=receipt_raw,
        expected_raw_batch_file_sha256=batch_file,
        expected_raw_batch_record_digest=batch_record,
        expected_recompute_receipt_file_sha256=receipt_file,
        expected_recompute_receipt_record_digest=receipt_record,
    )
    assert replay["pixel_reads"] == 0
    assert replay["feature_extraction_calls"] == 0
    assert replay["model_prediction_api_calls"] == 0
    assert replay["estimator_predict_proba_calls"] == 0
    assert replay["canonical_records_exact"] is True
    assert replay["recompute_receipt_join_exact"] is True


def test_tampering_extra_identity_fields_and_wrong_pixels_are_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_authority(monkeypatch)
    line, circle = _png("line"), _png("circle")
    kwargs = _creator_kwargs()
    batch = subject.create_raw_inference_batch(
        **kwargs, png_payloads=(line, line, circle)
    )
    raw = batch.to_data()
    raw["rows"][0]["role"] = "primary"
    with pytest.raises(subject.SkeletonGraphInferenceCustodyError, match="fields"):
        subject.SkeletonGraphRawInferenceBatch.from_data(raw)

    raw = batch.to_data()
    raw["rows"][0]["direct_pair_probabilities"][0] = -0.1
    row_body = dict(raw["rows"][0])
    row_body.pop("record_digest")
    raw["rows"][0] = subject._seal(row_body)
    batch_body = dict(raw)
    batch_body.pop("record_digest")
    with pytest.raises(
        subject.SkeletonGraphInferenceCustodyError, match="probability vector"
    ):
        subject.SkeletonGraphRawInferenceBatch.from_data(subject._seal(batch_body))

    with pytest.raises(
        subject.SkeletonGraphInferenceCustodyError, match="recomputation differs"
    ):
        subject.fresh_verify_raw_inference_batch(
            batch, **kwargs, png_payloads=(line, circle, circle)
        )


def test_archive_addresses_and_receipt_join_are_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_authority(monkeypatch)
    kwargs = _creator_kwargs()
    payload = _png("line")
    batch = subject.create_raw_inference_batch(**kwargs, png_payloads=(payload,))
    receipt = subject.fresh_verify_raw_inference_batch(
        batch, **kwargs, png_payloads=(payload,)
    )
    with pytest.raises(subject.SkeletonGraphInferenceCustodyError, match="issuance"):
        subject.SkeletonGraphInferenceRecomputeReceipt._issue_after_exact_recompute(
            batch, issuance_token=object()
        )
    batch_raw, batch_file, batch_record = _archive_addresses(batch)
    receipt_raw, receipt_file, receipt_record = _archive_addresses(receipt)
    with pytest.raises(subject.SkeletonGraphInferenceCustodyError, match="file address"):
        subject.cold_replay_raw_inference(
            raw_batch_bytes=batch_raw,
            recompute_receipt_bytes=receipt_raw + b" ",
            expected_raw_batch_file_sha256=batch_file,
            expected_raw_batch_record_digest=batch_record,
            expected_recompute_receipt_file_sha256=receipt_file,
            expected_recompute_receipt_record_digest=receipt_record,
        )

    other = subject.create_raw_inference_batch(
        **kwargs, png_payloads=(payload, _png("circle"))
    )
    other_raw, other_file, other_record = _archive_addresses(other)
    with pytest.raises(subject.SkeletonGraphInferenceCustodyError, match="join differs"):
        subject.cold_replay_raw_inference(
            raw_batch_bytes=other_raw,
            recompute_receipt_bytes=receipt_raw,
            expected_raw_batch_file_sha256=other_file,
            expected_raw_batch_record_digest=other_record,
            expected_recompute_receipt_file_sha256=receipt_file,
            expected_recompute_receipt_record_digest=receipt_record,
        )


def test_exact_passed_fit_protocol_is_required_before_any_pixel_decode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decoded = False

    def tripwire(_raw: bytes) -> np.ndarray:
        nonlocal decoded
        decoded = True
        raise AssertionError("pixel decoded before passed-fit rejection")

    monkeypatch.setattr(subject.core, "extract_feature_vector", tripwire)
    with pytest.raises(
        subject.SkeletonGraphInferenceCustodyError, match="exact passed-fit protocol"
    ):
        subject.create_raw_inference_batch(
            **_creator_kwargs(), png_payloads=(_png("line"),)
        )
    assert decoded is False


def test_canonical_replay_rejects_creator_cap_bypass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_authority(monkeypatch)
    payload = _png("line")
    batch = subject.create_raw_inference_batch(
        **_creator_kwargs(), png_payloads=(payload,)
    )
    raw = batch.to_data()
    row = dict(raw["rows"][0])
    row.pop("record_digest")
    row["occurrence_count"] = subject.MAX_INPUT_OCCURRENCES + 1
    raw["rows"][0] = subject._seal(row)
    raw.pop("record_digest")
    raw["input_occurrence_count"] = subject.MAX_INPUT_OCCURRENCES + 1
    raw["input_png_size_bytes"] = len(payload) * (
        subject.MAX_INPUT_OCCURRENCES + 1
    )
    with pytest.raises(
        subject.SkeletonGraphInferenceCustodyError,
        match="counts differ",
    ):
        subject.SkeletonGraphRawInferenceBatch.from_data(subject._seal(raw))

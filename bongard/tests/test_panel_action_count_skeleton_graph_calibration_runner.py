from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from bongard import panel_action_count_skeleton_graph_calibration_runner as subject
from bongard import panel_action_count_skeleton_graph_inference_custody as custody
from bongard.panel_action_count_skeleton_graph_passed_fit_protocol import (
    SkeletonGraphPassedFitGap,
    SkeletonGraphPassedFitProtocol,
)


ROOT = Path(__file__).resolve().parents[2]
PREREGISTRATION = (
    ROOT
    / "bongard"
    / "data"
    / "panel_action_count_skeleton_graph_calibration_preregistration_20260810_v1.json"
)
GENERIC_MANIFEST = (
    ROOT / "bongard" / "data" / "panel_action_count_cnn_calibration_panels_20260810_v3.json"
)
SHA_A = "sha256:" + "a" * 64
SHA_B = "sha256:" + "b" * 64


def _sha(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _reseal(path: Path, mutate) -> None:
    value = json.loads(path.read_bytes())
    value.pop("record_digest")
    mutate(value)
    path.write_bytes(subject.canonical_json(subject._seal(value)) + b"\n")


def _protocol() -> SkeletonGraphPassedFitProtocol:
    value = object.__new__(SkeletonGraphPassedFitProtocol)
    object.__setattr__(
        value, "record_digest", subject.PINNED_PASSED_FIT_PROTOCOL_RECORD_DIGEST
    )
    object.__setattr__(
        value,
        "passed_fit_authority_source_sha256",
        "sha256:" + subject.PINNED_PASSED_FIT_SOURCE_SHA256,
    )
    object.__setattr__(
        value,
        "passed_fit_algorithm_digest",
        subject.PINNED_PASSED_FIT_ALGORITHM_DIGEST,
    )
    return value


def _gap() -> SkeletonGraphPassedFitGap:
    value = object.__new__(SkeletonGraphPassedFitGap)
    object.__setattr__(value, "record_digest", SHA_B)
    return value


def _task_ids(scope: subject.SkeletonGraphCalibrationScope) -> tuple[str, ...]:
    if scope is subject.SkeletonGraphCalibrationScope.SAME_FAMILY:
        return tuple(subject.prereg.SAME_FAMILY_TASK_IDS)
    manifest = json.loads(GENERIC_MANIFEST.read_bytes())
    return tuple(manifest["cohorts"]["calibration"]["task_ids"])


PRIMARY_PAYLOAD = b"synthetic-primary-calibration-payload"
CONTRAST_PAYLOAD = b"synthetic-contrast-calibration-payload"


def _identities(
    scope: subject.SkeletonGraphCalibrationScope,
) -> tuple[subject.SkeletonGraphCalibrationPanelIdentity, ...]:
    result = []
    for task_id in _task_ids(scope):
        for side in (1, 0):
            payload = PRIMARY_PAYLOAD if side == 1 else CONTRAST_PAYLOAD
            for ordinal in range(7):
                result.append(
                    subject.SkeletonGraphCalibrationPanelIdentity(
                        panel_id=f"hd/{task_id}/{side}/{ordinal}.png",
                        png_sha256=_sha(payload),
                        png_size=len(payload),
                    )
                )
    return tuple(result)


def _probabilities(code: int) -> np.ndarray:
    result = np.full(33, 0.01 / 32.0, dtype=np.float64)
    result[tuple(custody.DIRECT_PAIR_CLASS_ORDER).index(code)] = 0.99
    return result


def _catalog(value: int) -> np.ndarray:
    result = np.full(3, 0.005, dtype=np.float64)
    result[tuple(custody.CATALOG_CLASS_ORDER).index(value)] = 0.99
    return result


def _inference_pair(
    payloads: tuple[bytes, ...], calls: list[str]
) -> tuple[custody.SkeletonGraphRawInferenceBatch, custody.SkeletonGraphInferenceRecomputeReceipt]:
    calls.append("inference")
    counts = Counter(payloads)
    bindings = {
        "core_source_sha256": custody.core.source_sha256(),
        "core_config_digest": custody.core.config_digest(),
        "model_file_sha256": subject.prereg.MODEL_FILE_SHA256,
        "passed_fit_protocol_record_digest": (
            subject.PINNED_PASSED_FIT_PROTOCOL_RECORD_DIGEST
        ),
        "passed_fit_authority_source_sha256": (
            "sha256:" + custody.passed_fit_module.source_sha256()
        ),
        "passed_fit_algorithm_digest": custody.passed_fit_module.PASSED_FIT_ALGORITHM_DIGEST,
        "inference_source_sha256": custody.source_sha256(),
        "inference_algorithm_digest": custody.algorithm_digest(),
    }
    rows = []
    for payload, pair_code, catalog_class in (
        (PRIMARY_PAYLOAD, 40, 1),
        (CONTRAST_PAYLOAD, 20, 0),
    ):
        rows.append(
            custody.SkeletonGraphRawInferenceRow.from_arrays(
                png_sha256=_sha(payload),
                png_size_bytes=len(payload),
                occurrence_count=counts[payload],
                feature=np.zeros(112, dtype=np.float32),
                direct_pair_probabilities=_probabilities(pair_code),
                catalog_probabilities=_catalog(catalog_class),
                bindings=bindings,
            )
        )
    batch = custody.SkeletonGraphRawInferenceBatch.from_rows(
        sorted(rows, key=lambda row: row.png_sha256)
    )
    receipt = custody.SkeletonGraphInferenceRecomputeReceipt._issue_after_exact_recompute(
        batch, issuance_token=custody._RECOMPUTE_ISSUANCE_TOKEN
    )
    return batch, receipt


def _paths() -> subject.SkeletonGraphPassedFitPaths:
    unused = Path("synthetic-unused")
    return subject.SkeletonGraphPassedFitPaths(
        unused, unused, unused, unused, unused, unused
    )


def _install_synthetic_metadata_boundary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    outcome: SkeletonGraphPassedFitProtocol | SkeletonGraphPassedFitGap,
) -> Path:
    monkeypatch.setattr(subject, "_verify_passed_fit_outcome", lambda *_args: outcome)
    authority = tmp_path / "single-global-attempt-authority"
    authority.mkdir()
    registration = subject._load_preregistration(PREREGISTRATION)
    predecessor_path = ROOT / registration["exposure_predecessor"]["ledger_path"]
    predecessor_raw = predecessor_path.read_bytes()
    predecessor_name = predecessor_path.name
    (authority / predecessor_name).write_bytes(predecessor_raw)

    def campaign_authority(_registration, _scope):
        fresh_predecessor_raw = (authority / predecessor_name).read_bytes()
        return subject._CampaignAuthority(
            parent=subject._existing_output_directory(authority),
            predecessor=subject._decode_exposure_ledger(
                fresh_predecessor_raw, label="synthetic predecessor"
            ),
            predecessor_raw=fresh_predecessor_raw,
            predecessor_filename=predecessor_name,
            intent_filename="panel_action_count_skeleton_graph_campaign_attempt_v2.json",
        )

    monkeypatch.setattr(
        subject,
        "_campaign_attempt_authority",
        campaign_authority,
    )
    return authority


def _callbacks(
    output: Path, calls: list[str]
):
    def pixels(panel_id: str) -> bytes:
        calls.append("pixel")
        return PRIMARY_PAYLOAD if "/1/" in panel_id else CONTRAST_PAYLOAD

    def infer(payloads, passed_fit):
        assert type(passed_fit) is SkeletonGraphPassedFitProtocol
        assert all(type(payload) is bytes for payload in payloads)
        return _inference_pair(payloads, calls)

    def label_factory():
        calls.append("label_factory")
        assert (output / "raw_predictions.json").is_file()

        def read(request: subject.SkeletonGraphDelayedLabelRequest):
            calls.append("labels")
            subject.verify_and_consume_delayed_label_request(request)
            rows = []
            for token, identity in request.bindings:
                rows.append(
                    subject.SkeletonGraphDelayedLabelRow(
                        anonymous_panel_token=token,
                        panel_id=identity.panel_id,
                        task_id=identity.task_id,
                        side=identity.side,
                        ordinal=identity.ordinal,
                        true_straight_action_count=4 if identity.side == 1 else 2,
                        true_arc_action_count=0,
                        true_catalog_class=1 if identity.side == 1 else 0,
                    )
                )
            return subject.SkeletonGraphDelayedLabelBatch.create(
                delayed_label_request_record_digest=request.record_digest,
                label_attempt_record_digest=request.label_attempt_record_digest,
                prediction_record_digest=request.prediction_record_digest,
                prediction_file_sha256=request.prediction_file_sha256,
                action_program_authority_record_digest=SHA_A,
                action_program_authority_file_sha256=SHA_B,
                catalog_authority_source_sha256=SHA_A,
                catalog_algorithm_digest=SHA_B,
                label_extraction_algorithm_digest=_sha(b"synthetic-label-deriver"),
                rows=rows,
            )

        return read

    return pixels, infer, label_factory


def _authorize_and_run_success(
    *,
    scope: subject.SkeletonGraphCalibrationScope,
    protocol: SkeletonGraphPassedFitProtocol,
    output: Path,
) -> tuple[
    subject.SkeletonGraphCalibrationExposureAuthorization,
    subject.SkeletonGraphPopulationGrant,
    list[str],
]:
    calls: list[str] = []
    pixels, infer, labels = _callbacks(output, calls)
    authorization = subject.authorize_calibration_exposure(
        scope=scope,
        preregistration_path=PREREGISTRATION,
        passed_fit=protocol,
        passed_fit_paths=_paths(),
        output_directory=output,
    )
    assert type(authorization) is subject.SkeletonGraphCalibrationExposureAuthorization
    grant = subject.run_calibration(
        exposure_authorization=authorization,
        preregistration_path=PREREGISTRATION,
        passed_fit=protocol,
        passed_fit_paths=_paths(),
        panel_identities=_identities(scope),
        output_directory=output,
        calibration_pixel_reader=pixels,
        inference_runner=infer,
        delayed_label_reader_factory=labels,
    )
    assert type(grant) is subject.SkeletonGraphPopulationGrant
    return authorization, grant, calls


@pytest.mark.parametrize(
    "scope",
    [
        subject.SkeletonGraphCalibrationScope.GENERIC_V3,
        subject.SkeletonGraphCalibrationScope.SAME_FAMILY,
    ],
)
def test_synthetic_archive_is_causal_replayable_and_explicitly_not_production(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    scope: subject.SkeletonGraphCalibrationScope,
) -> None:
    protocol = _protocol()
    _install_synthetic_metadata_boundary(monkeypatch, tmp_path, protocol)
    output = tmp_path / "run"
    calls: list[str] = []
    pixels, infer, labels = _callbacks(output, calls)
    authorization = subject.authorize_calibration_exposure(
        scope=scope,
        preregistration_path=PREREGISTRATION,
        passed_fit=protocol,
        passed_fit_paths=_paths(),
        output_directory=output,
    )
    assert type(authorization) is subject.SkeletonGraphCalibrationExposureAuthorization
    grant = subject.run_calibration(
        exposure_authorization=authorization,
        preregistration_path=PREREGISTRATION,
        passed_fit=protocol,
        passed_fit_paths=_paths(),
        panel_identities=_identities(scope),
        output_directory=output,
        calibration_pixel_reader=pixels,
        inference_runner=infer,
        delayed_label_reader_factory=labels,
    )
    assert type(grant) is subject.SkeletonGraphPopulationGrant
    assert calls.count("inference") == 1
    assert calls.index("inference") < calls.index("label_factory") < calls.index("labels")
    assert grant.authenticated_calibration_execution is False
    assert grant.production_adapter_authorized is False
    assert grant.target_pixel_authorized is False
    assert grant.authorizes_task(grant.calibration_task_ids[0])
    if scope is subject.SkeletonGraphCalibrationScope.GENERIC_V3:
        assert not grant.authorizes_target_scope(subject.SAME_FAMILY_TARGET_TASK_ID)
        assert not grant.authorizes_task("arbitrary-generic-task")
    else:
        assert grant.efficiency_gate["all_fixed_checks_passed"] is True
        assert grant.authorizes_target_scope(subject.SAME_FAMILY_TARGET_TASK_ID)

    replay = subject.cold_replay_calibration(
        run_directory=output, preregistration_path=PREREGISTRATION
    )
    assert type(replay) is subject.SkeletonGraphCalibrationReplayReceipt
    assert replay.verifies(grant)
    assert replay.authenticated_calibration_execution is False
    assert replay.production_adapter_authorized is False
    assert subject.verify_skeleton_graph_population_grant(
        grant, replay_receipt=replay
    ) is grant
    replay_again = subject.cold_replay_calibration(
        run_directory=output, preregistration_path=PREREGISTRATION
    )
    assert replay_again.to_data() == replay.to_data()


def test_passed_fit_gap_opens_no_callbacks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    gap = _gap()
    authority = _install_synthetic_metadata_boundary(monkeypatch, tmp_path, gap)
    calls: list[str] = []

    def forbidden(*_args, **_kwargs):
        calls.append("forbidden")
        raise AssertionError("callback opened")

    result = subject.authorize_calibration_exposure(
        scope=subject.SkeletonGraphCalibrationScope.GENERIC_V3,
        preregistration_path=PREREGISTRATION,
        passed_fit=gap,
        passed_fit_paths=_paths(),
        output_directory=tmp_path / "gap-run",
    )
    assert type(result) is subject.SkeletonGraphCalibrationGap
    assert result.stage == "passed_fit_precommit"
    assert calls == []
    assert not (tmp_path / "gap-run").exists()
    assert [path.name for path in authority.iterdir()] == [
        "6995ea9cfda2f384cb0ba1b1cdc3611c965227c60fdb281d1e2e56fffa357b56.exposure.json"
    ]


def test_partial_inference_never_opens_label_authority_and_attempt_cannot_reroll(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    protocol = _protocol()
    _install_synthetic_metadata_boundary(monkeypatch, tmp_path, protocol)
    scope = subject.SkeletonGraphCalibrationScope.SAME_FAMILY
    output = tmp_path / "partial"
    labels_opened = 0

    def pixels(panel_id: str) -> bytes:
        return PRIMARY_PAYLOAD if "/1/" in panel_id else CONTRAST_PAYLOAD

    def label_factory():
        nonlocal labels_opened
        labels_opened += 1
        raise AssertionError("label authority opened")

    authorization = subject.authorize_calibration_exposure(
        scope=scope,
        preregistration_path=PREREGISTRATION,
        passed_fit=protocol,
        passed_fit_paths=_paths(),
        output_directory=output,
    )
    assert type(authorization) is subject.SkeletonGraphCalibrationExposureAuthorization
    with pytest.raises(subject.SkeletonGraphCalibrationRunnerError):
        subject.run_calibration(
            exposure_authorization=authorization,
            preregistration_path=PREREGISTRATION,
            passed_fit=protocol,
            passed_fit_paths=_paths(),
            panel_identities=_identities(scope),
            output_directory=output,
            calibration_pixel_reader=pixels,
            inference_runner=lambda *_args: (object(),),
            delayed_label_reader_factory=label_factory,
        )
    assert labels_opened == 0
    assert (output / "prediction_attempt.json").is_file()
    assert (output / "calibration_gap.json").is_file()
    replayed = subject.cold_replay_calibration(
        run_directory=output, preregistration_path=PREREGISTRATION
    )
    assert type(replayed) is subject.SkeletonGraphCalibrationGap
    assert replayed.stage == "integrity_pixel_or_inference_callback"
    with pytest.raises(subject.SkeletonGraphCalibrationRunnerError):
        subject.authorize_calibration_exposure(
            scope=scope,
            preregistration_path=PREREGISTRATION,
            passed_fit=protocol,
            passed_fit_paths=_paths(),
            output_directory=tmp_path / "alternate-root",
        )
    assert not (tmp_path / "alternate-root").exists()


def test_output_and_global_parent_substitution_fail_closed(tmp_path: Path) -> None:
    for name in ("output", "global-parent"):
        original = tmp_path / name
        original.mkdir()
        custody_value = subject._existing_output_directory(original)
        moved = tmp_path / f"{name}-moved"
        original.rename(moved)
        original.symlink_to(moved, target_is_directory=True)
        with pytest.raises(
            subject.SkeletonGraphCalibrationRunnerError,
            match="renamed, replaced, or redirected",
        ):
            subject._write_output_record(
                custody_value,
                "must-not-write.json",
                {"schema": "gkm.test-output-root-substitution.v1"},
            )
        assert not (moved / "must-not-write.json").exists()


def test_real_metadata_attempt_path_authenticates_predecessor_without_writes() -> None:
    registration = subject._load_preregistration(PREREGISTRATION)
    authority = subject._campaign_attempt_authority(
        registration, subject.SkeletonGraphCalibrationScope.GENERIC_V3
    )
    try:
        assert authority.parent.path.name == "research-exposure-successors"
        assert authority.intent_filename == (
            "panel_action_count_skeleton_graph_campaign_attempt_v2.json"
        )
        assert registration["exposure_predecessor"]["ledger_source_sha256"] == (
            "sha256:8c5034e77f769a67b1bc16b41881e14887592e070e730d062049ea33e1467ff8"
        )
    finally:
        authority.close()


def test_metadata_authorization_appends_exact_ledger_before_pixels_and_reuses_intent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    protocol = _protocol()
    authority_dir = _install_synthetic_metadata_boundary(
        monkeypatch, tmp_path, protocol
    )
    scope = subject.SkeletonGraphCalibrationScope.SAME_FAMILY
    output = tmp_path / "chronology-run"
    clock_calls = 0

    def clock() -> str:
        nonlocal clock_calls
        clock_calls += 1
        if clock_calls != 1:
            raise AssertionError("intent recovery called the clock")
        return "2026-08-10T12:34:56.000001Z"

    monkeypatch.setattr(subject, "_new_observed_at", clock)
    authorization = subject.authorize_calibration_exposure(
        scope=scope,
        preregistration_path=PREREGISTRATION,
        passed_fit=protocol,
        passed_fit_paths=_paths(),
        output_directory=output,
    )
    assert type(authorization) is subject.SkeletonGraphCalibrationExposureAuthorization
    assert not output.exists()
    predecessor = subject._decode_exposure_ledger(
        (authority_dir / authorization.exposure_predecessor_filename).read_bytes(),
        label="test predecessor",
    )
    successor_path = authority_dir / authorization.exposure_successor_filename
    successor = subject._decode_exposure_ledger(
        successor_path.read_bytes(), label="test successor"
    )
    assert len(successor.events) == len(predecessor.events) + 1
    assert successor.events[:-1] == predecessor.events
    event = successor.events[-1]
    assert event.task_ids == tuple(sorted(authorization.task_ids))
    assert event.panel_ids == tuple(sorted(authorization.panel_ids))
    assert event.observed_at == "2026-08-10T12:34:56.000001Z"
    assert event.digest == authorization.exposure_event_digest
    assert _sha(successor_path.read_bytes()) == authorization.exposure_successor_file_sha256

    recovered = subject.authorize_calibration_exposure(
        scope=scope,
        preregistration_path=PREREGISTRATION,
        passed_fit=protocol,
        passed_fit_paths=_paths(),
        output_directory=output,
    )
    assert recovered.to_data() == authorization.to_data()
    assert clock_calls == 1

    calls: list[str] = []
    pixels, infer, labels = _callbacks(output, calls)

    def guarded_pixels(panel_id: str) -> bytes:
        assert successor_path.is_file()
        assert subject._decode_exposure_ledger(
            successor_path.read_bytes(), label="pixel-bound successor"
        ).digest == authorization.exposure_successor_ledger_digest
        return pixels(panel_id)

    result = subject.run_calibration(
        exposure_authorization=authorization,
        preregistration_path=PREREGISTRATION,
        passed_fit=protocol,
        passed_fit_paths=_paths(),
        panel_identities=_identities(scope),
        output_directory=output,
        calibration_pixel_reader=guarded_pixels,
        inference_runner=infer,
        delayed_label_reader_factory=labels,
    )
    assert type(result) is subject.SkeletonGraphPopulationGrant
    assert calls.count("pixel") == 224


def test_crash_after_intent_recovers_only_exact_child_without_new_time(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    protocol = _protocol()
    authority_dir = _install_synthetic_metadata_boundary(
        monkeypatch, tmp_path, protocol
    )
    scope = subject.SkeletonGraphCalibrationScope.SAME_FAMILY
    output = tmp_path / "crash-recovery"
    monkeypatch.setattr(
        subject, "_new_observed_at", lambda: "2026-08-10T00:00:01.000000Z"
    )
    original_persist = subject._persist_exposure_successor
    failed = False

    def fail_once(*args, **kwargs):
        nonlocal failed
        if not failed:
            failed = True
            raise subject.SkeletonGraphCalibrationRunnerError("synthetic crash")
        return original_persist(*args, **kwargs)

    monkeypatch.setattr(subject, "_persist_exposure_successor", fail_once)
    with pytest.raises(subject.SkeletonGraphCalibrationRunnerError, match="synthetic crash"):
        subject.authorize_calibration_exposure(
            scope=scope,
            preregistration_path=PREREGISTRATION,
            passed_fit=protocol,
            passed_fit_paths=_paths(),
            output_directory=output,
        )
    intent_path = authority_dir / "panel_action_count_skeleton_graph_campaign_attempt_v2.json"
    intent_before = intent_path.read_bytes()
    assert not list(authority_dir.glob("*.calibration-authorization.json"))
    monkeypatch.setattr(subject, "_persist_exposure_successor", original_persist)
    monkeypatch.setattr(
        subject,
        "_new_observed_at",
        lambda: (_ for _ in ()).throw(AssertionError("recovery regenerated time")),
    )
    authorization = subject.authorize_calibration_exposure(
        scope=scope,
        preregistration_path=PREREGISTRATION,
        passed_fit=protocol,
        passed_fit_paths=_paths(),
        output_directory=output,
    )
    assert intent_path.read_bytes() == intent_before
    assert authorization.exposure_event_observed_at == "2026-08-10T00:00:01.000000Z"


def test_deleted_intent_cannot_reroll_or_fork_successor(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    protocol = _protocol()
    authority_dir = _install_synthetic_metadata_boundary(
        monkeypatch, tmp_path, protocol
    )
    output = tmp_path / "deleted-intent"
    authorization = subject.authorize_calibration_exposure(
        scope=subject.SkeletonGraphCalibrationScope.SAME_FAMILY,
        preregistration_path=PREREGISTRATION,
        passed_fit=protocol,
        passed_fit_paths=_paths(),
        output_directory=output,
    )
    exposure_names = sorted(path.name for path in authority_dir.glob("*.exposure.json"))
    (authority_dir / authorization.campaign_intent_filename).unlink()
    (authority_dir / authorization.filename).unlink()
    with pytest.raises(
        subject.SkeletonGraphCalibrationRunnerError,
        match="missing beside an issued authorization",
    ):
        subject.authorize_calibration_exposure(
            scope=subject.SkeletonGraphCalibrationScope.SAME_FAMILY,
            preregistration_path=PREREGISTRATION,
            passed_fit=protocol,
            passed_fit_paths=_paths(),
            output_directory=output,
        )
    assert sorted(path.name for path in authority_dir.glob("*.exposure.json")) == exposure_names


def test_authorized_output_parent_swap_fails_before_callbacks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    protocol = _protocol()
    _install_synthetic_metadata_boundary(monkeypatch, tmp_path, protocol)
    output_parent = tmp_path / "output-parent"
    output_parent.mkdir()
    output = output_parent / "run"
    authorization = subject.authorize_calibration_exposure(
        scope=subject.SkeletonGraphCalibrationScope.SAME_FAMILY,
        preregistration_path=PREREGISTRATION,
        passed_fit=protocol,
        passed_fit_paths=_paths(),
        output_directory=output,
    )
    moved = tmp_path / "output-parent-moved"
    output_parent.rename(moved)
    output_parent.symlink_to(moved, target_is_directory=True)
    calls: list[str] = []

    def forbidden(*_args, **_kwargs):
        calls.append("opened")
        raise AssertionError("callback opened after parent swap")

    with pytest.raises(subject.SkeletonGraphCalibrationRunnerError):
        subject.run_calibration(
            exposure_authorization=authorization,
            preregistration_path=PREREGISTRATION,
            passed_fit=protocol,
            passed_fit_paths=_paths(),
            panel_identities=_identities(subject.SkeletonGraphCalibrationScope.SAME_FAMILY),
            output_directory=output,
            calibration_pixel_reader=forbidden,
            inference_runner=forbidden,
            delayed_label_reader_factory=forbidden,
        )
    assert calls == []
    assert not (moved / "run").exists()


@pytest.mark.parametrize("attempt_name", ["prediction_attempt.json", "label_attempt.json"])
def test_cold_replay_rejects_deleted_attempt_custody(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, attempt_name: str
) -> None:
    protocol = _protocol()
    _install_synthetic_metadata_boundary(monkeypatch, tmp_path, protocol)
    output = tmp_path / "attempt-deletion"
    _authorize_and_run_success(
        scope=subject.SkeletonGraphCalibrationScope.SAME_FAMILY,
        protocol=protocol,
        output=output,
    )
    (output / attempt_name).unlink()
    with pytest.raises(subject.SkeletonGraphCalibrationRunnerError):
        subject.cold_replay_calibration(
            run_directory=output, preregistration_path=PREREGISTRATION
        )


def test_cold_replay_rejects_resealed_bool_attempt_tamper(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    protocol = _protocol()
    _install_synthetic_metadata_boundary(monkeypatch, tmp_path, protocol)
    output = tmp_path / "attempt-tamper"
    _authorize_and_run_success(
        scope=subject.SkeletonGraphCalibrationScope.SAME_FAMILY,
        protocol=protocol,
        output=output,
    )
    path = output / "label_attempt.json"
    value = json.loads(path.read_bytes())
    value.pop("record_digest")
    value["attempt_number"] = True
    path.write_bytes(subject.canonical_json(subject._seal(value)) + b"\n")
    with pytest.raises(subject.SkeletonGraphCalibrationRunnerError):
        subject.cold_replay_calibration(
            run_directory=output, preregistration_path=PREREGISTRATION
        )


def test_concurrent_cold_replay_has_one_identical_winner(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    protocol = _protocol()
    _install_synthetic_metadata_boundary(monkeypatch, tmp_path, protocol)
    output = tmp_path / "replay-race"
    _, grant, _ = _authorize_and_run_success(
        scope=subject.SkeletonGraphCalibrationScope.SAME_FAMILY,
        protocol=protocol,
        output=output,
    )

    def replay():
        return subject.cold_replay_calibration(
            run_directory=output, preregistration_path=PREREGISTRATION
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        first, second = tuple(pool.map(lambda _index: replay(), range(2)))
    assert type(first) is subject.SkeletonGraphCalibrationReplayReceipt
    assert first.to_data() == second.to_data()
    assert first.verifies(grant)


def test_write_failure_terminalizes_once_and_replays_without_labels(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    protocol = _protocol()
    _install_synthetic_metadata_boundary(monkeypatch, tmp_path, protocol)
    scope = subject.SkeletonGraphCalibrationScope.SAME_FAMILY
    output = tmp_path / "write-failure"
    authorization = subject.authorize_calibration_exposure(
        scope=scope,
        preregistration_path=PREREGISTRATION,
        passed_fit=protocol,
        passed_fit_paths=_paths(),
        output_directory=output,
    )
    calls: list[str] = []
    pixels, infer, labels = _callbacks(output, calls)
    original_write = subject._write_output_record
    failed = False

    def fail_prediction(custody_value, name, body):
        nonlocal failed
        if name == "raw_predictions.json" and not failed:
            failed = True
            raise subject.SkeletonGraphCalibrationRunnerError("synthetic write failure")
        return original_write(custody_value, name, body)

    monkeypatch.setattr(subject, "_write_output_record", fail_prediction)
    with pytest.raises(subject.SkeletonGraphCalibrationRunnerError, match="synthetic write"):
        subject.run_calibration(
            exposure_authorization=authorization,
            preregistration_path=PREREGISTRATION,
            passed_fit=protocol,
            passed_fit_paths=_paths(),
            panel_identities=_identities(scope),
            output_directory=output,
            calibration_pixel_reader=pixels,
            inference_runner=infer,
            delayed_label_reader_factory=labels,
        )
    assert (output / "calibration_gap.json").is_file()
    assert not (output / "delayed_labels.json").exists()
    assert "label_factory" not in calls
    replayed = subject.cold_replay_calibration(
        run_directory=output, preregistration_path=PREREGISTRATION
    )
    assert type(replayed) is subject.SkeletonGraphCalibrationGap
    assert replayed.stage == "integrity_prediction_write"


def test_delayed_label_request_is_factory_only_tamper_evident_and_one_shot(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with pytest.raises(subject.SkeletonGraphCalibrationRunnerError):
        subject.SkeletonGraphDelayedLabelRequest()
    protocol = _protocol()
    _install_synthetic_metadata_boundary(monkeypatch, tmp_path, protocol)
    scope = subject.SkeletonGraphCalibrationScope.SAME_FAMILY
    output = tmp_path / "label-capability"
    authorization = subject.authorize_calibration_exposure(
        scope=scope,
        preregistration_path=PREREGISTRATION,
        passed_fit=protocol,
        passed_fit_paths=_paths(),
        output_directory=output,
    )
    captured: list[subject.SkeletonGraphDelayedLabelRequest] = []
    calls: list[str] = []
    pixels, infer, base_factory = _callbacks(output, calls)

    def factory():
        base_reader = base_factory()

        def reader(request):
            captured.append(request)
            forged = copy.copy(request)
            object.__setattr__(forged, "prediction_record_digest", SHA_A)
            with pytest.raises(subject.SkeletonGraphCalibrationRunnerError):
                subject.verify_and_consume_delayed_label_request(forged)
            return base_reader(request)

        return reader

    result = subject.run_calibration(
        exposure_authorization=authorization,
        preregistration_path=PREREGISTRATION,
        passed_fit=protocol,
        passed_fit_paths=_paths(),
        panel_identities=_identities(scope),
        output_directory=output,
        calibration_pixel_reader=pixels,
        inference_runner=infer,
        delayed_label_reader_factory=factory,
    )
    assert type(result) is subject.SkeletonGraphPopulationGrant
    assert len(captured) == 1
    with pytest.raises(
        subject.SkeletonGraphCalibrationRunnerError, match="already consumed"
    ):
        subject.verify_and_consume_delayed_label_request(captured[0])


def test_core_rejects_label_reader_that_does_not_consume_sealed_request(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    protocol = _protocol()
    _install_synthetic_metadata_boundary(monkeypatch, tmp_path, protocol)
    scope = subject.SkeletonGraphCalibrationScope.SAME_FAMILY
    output = tmp_path / "unconsumed-label-request"
    authorization = subject.authorize_calibration_exposure(
        scope=scope,
        preregistration_path=PREREGISTRATION,
        passed_fit=protocol,
        passed_fit_paths=_paths(),
        output_directory=output,
    )
    calls: list[str] = []
    pixels, infer, labels = _callbacks(output, calls)
    monkeypatch.setattr(
        subject, "verify_and_consume_delayed_label_request", lambda request: request
    )
    with pytest.raises(
        subject.SkeletonGraphCalibrationRunnerError, match="did not consume"
    ):
        subject.run_calibration(
            exposure_authorization=authorization,
            preregistration_path=PREREGISTRATION,
            passed_fit=protocol,
            passed_fit_paths=_paths(),
            panel_identities=_identities(scope),
            output_directory=output,
            calibration_pixel_reader=pixels,
            inference_runner=infer,
            delayed_label_reader_factory=labels,
        )
    gap = json.loads((output / "calibration_gap.json").read_bytes())
    assert gap["stage"] == "integrity_delayed_label_callback"


def test_bool_values_are_rejected_for_integer_fields(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with pytest.raises(subject.SkeletonGraphCalibrationRunnerError):
        subject.SkeletonGraphCalibrationPanelIdentity(
            "hd/task/1/0.png", SHA_A, True
        )
    with pytest.raises(subject.SkeletonGraphCalibrationRunnerError):
        subject.SkeletonGraphDelayedLabelRow(
            anonymous_panel_token="anon_" + "0" * 64,
            panel_id="hd/task/1/0.png",
            task_id="task",
            side=True,
            ordinal=0,
            true_straight_action_count=4,
            true_arc_action_count=0,
            true_catalog_class=True,
        )
    protocol = _protocol()
    _install_synthetic_metadata_boundary(monkeypatch, tmp_path, protocol)
    authorization = subject.authorize_calibration_exposure(
        scope=subject.SkeletonGraphCalibrationScope.SAME_FAMILY,
        preregistration_path=PREREGISTRATION,
        passed_fit=protocol,
        passed_fit_paths=_paths(),
        output_directory=tmp_path / "bool-auth",
    )
    wire = authorization.to_data()
    wire["intended_output_parent_st_dev"] = True
    body = dict(wire)
    body.pop("record_digest")
    wire["record_digest"] = "sha256:" + subject.canonical_digest(body)
    with pytest.raises(subject.SkeletonGraphCalibrationRunnerError):
        subject._authorization_from_data(wire)

    ledger = subject.ExposureLedger.create(SHA_A).record(
        phase="test",
        actor="test",
        purpose="test",
        task_ids=("hd_test",),
        observed_at="2026-08-10T00:00:00Z",
    )
    bool_sequence = ledger.to_json().encode().replace(
        b'"sequence": 0', b'"sequence": false', 1
    )
    with pytest.raises(
        subject.SkeletonGraphCalibrationRunnerError,
        match="non-exact integer event sequence",
    ):
        subject._decode_exposure_ledger(bool_sequence, label="bool ledger")


def test_production_successor_types_are_absent_from_core() -> None:
    assert not hasattr(subject, "SkeletonGraphAuthenticatedCalibrationExecutionReceipt")
    assert not hasattr(subject, "SkeletonGraphVerifiedAuthenticatedCalibrationExecution")
    assert not hasattr(subject, "SkeletonGraphExternalPopulationAuthorization")
    assert not hasattr(subject, "EXTERNAL_POPULATION_AUTHORIZATION_SCHEMA")


def test_torn_unique_private_temp_does_not_block_retry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    directory = tmp_path / "atomic"
    directory.mkdir(mode=0o700)
    held = subject._existing_output_directory(directory)
    raw = b'{"complete":true}\n'
    real_write = subject.os.write
    real_unlink = subject.os.unlink
    writes = 0

    def torn_write(descriptor, payload):
        nonlocal writes
        writes += 1
        if writes == 1:
            return real_write(descriptor, payload[:3])
        raise OSError("synthetic torn private inode")

    def leave_private(name, *args, **kwargs):
        if ".artifact.json.pending." in str(name):
            raise OSError("synthetic cleanup crash")
        return real_unlink(name, *args, **kwargs)

    try:
        monkeypatch.setattr(subject.os, "write", torn_write)
        monkeypatch.setattr(subject.os, "unlink", leave_private)
        with pytest.raises(subject.SkeletonGraphCalibrationRunnerError):
            subject._atomic_write_once_bytes(
                held,
                "artifact.json",
                raw,
                label="atomic test artifact",
                allow_identical_existing=False,
            )
        pending = [item.name for item in directory.iterdir()]
        assert len(pending) == 1
        assert pending[0] != ".artifact.json.pending"
        assert ".artifact.json.pending." in pending[0]

        monkeypatch.setattr(subject.os, "write", real_write)
        monkeypatch.setattr(subject.os, "unlink", real_unlink)
        assert subject._atomic_write_once_bytes(
            held,
            "artifact.json",
            raw,
            label="atomic test artifact",
            allow_identical_existing=False,
        ) == raw
        assert (directory / "artifact.json").read_bytes() == raw
    finally:
        held.close()


@pytest.mark.parametrize("fault", ["directory_fsync", "fresh_read", "cleanup"])
def test_exact_publication_survives_post_link_fault(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fault: str
) -> None:
    directory = tmp_path / f"post-link-{fault}"
    directory.mkdir(mode=0o700)
    held = subject._existing_output_directory(directory)
    raw = b'{"complete":true}\n'
    linked = False
    fired = False
    real_link = subject.os.link
    real_fsync = subject.os.fsync
    real_fstat = subject.os.fstat
    real_read = subject._read_dirfd_bytes
    real_unlink = subject.os.unlink

    def link(*args, **kwargs):
        nonlocal linked
        result = real_link(*args, **kwargs)
        linked = True
        return result

    def fsync(descriptor):
        nonlocal fired
        if (
            fault == "directory_fsync"
            and linked
            and not fired
            and subject.stat.S_ISDIR(real_fstat(descriptor).st_mode)
        ):
            fired = True
            raise OSError("synthetic post-link directory fsync fault")
        return real_fsync(descriptor)

    def read(descriptor, name, **kwargs):
        nonlocal fired
        if fault == "fresh_read" and linked and not fired and name == "artifact.json":
            fired = True
            raise subject.SkeletonGraphCalibrationRunnerError(
                "synthetic post-link reload fault"
            )
        return real_read(descriptor, name, **kwargs)

    def unlink(name, *args, **kwargs):
        nonlocal fired
        if fault == "cleanup" and linked and not fired and ".pending." in str(name):
            fired = True
            raise OSError("synthetic private cleanup fault")
        return real_unlink(name, *args, **kwargs)

    try:
        monkeypatch.setattr(subject.os, "link", link)
        monkeypatch.setattr(subject.os, "fsync", fsync)
        monkeypatch.setattr(subject, "_read_dirfd_bytes", read)
        monkeypatch.setattr(subject.os, "unlink", unlink)
        assert subject._atomic_write_once_bytes(
            held,
            "artifact.json",
            raw,
            label="post-link artifact",
            allow_identical_existing=False,
        ) == raw
        assert fired is True
        assert (directory / "artifact.json").read_bytes() == raw
    finally:
        held.close()


def test_persistent_directory_fsync_failure_requires_later_exact_retry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    directory = tmp_path / "persistent-dir-fsync"
    directory.mkdir(mode=0o700)
    held = subject._existing_output_directory(directory)
    raw = b'{"complete":true}\n'
    real_fsync = subject.os.fsync
    real_fstat = subject.os.fstat

    def fail_directory_fsync(descriptor):
        if subject.stat.S_ISDIR(real_fstat(descriptor).st_mode):
            raise OSError("synthetic persistent directory fsync failure")
        return real_fsync(descriptor)

    try:
        monkeypatch.setattr(subject.os, "fsync", fail_directory_fsync)
        with pytest.raises(
            subject.SkeletonGraphCalibrationRunnerError,
            match="persistent directory fsync",
        ):
            subject._atomic_write_once_bytes(
                held,
                "artifact.json",
                raw,
                label="persistent-fsync artifact",
                allow_identical_existing=True,
            )
        assert (directory / "artifact.json").read_bytes() == raw

        monkeypatch.setattr(subject.os, "fsync", real_fsync)
        assert subject._atomic_write_once_bytes(
            held,
            "artifact.json",
            raw,
            label="persistent-fsync artifact",
            allow_identical_existing=True,
        ) == raw
    finally:
        held.close()


@pytest.mark.parametrize(
    ("target", "expected_stage"),
    [
        ("authorization.json", "integrity_execution_authorization"),
        ("precommit.json", "integrity_precommit"),
    ],
)
def test_early_write_failure_terminal_gap_cold_replays_without_base_pair(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    target: str,
    expected_stage: str,
) -> None:
    protocol = _protocol()
    _install_synthetic_metadata_boundary(monkeypatch, tmp_path, protocol)
    scope = subject.SkeletonGraphCalibrationScope.SAME_FAMILY
    output = tmp_path / target.removesuffix(".json")
    authorization = subject.authorize_calibration_exposure(
        scope=scope,
        preregistration_path=PREREGISTRATION,
        passed_fit=protocol,
        passed_fit_paths=_paths(),
        output_directory=output,
    )
    assert type(authorization) is subject.SkeletonGraphCalibrationExposureAuthorization
    calls: list[str] = []
    pixels, infer, labels = _callbacks(output, calls)
    real_write = subject._write_output_record

    def fail_before(custody_value, name, body):
        if name == target:
            raise subject.SkeletonGraphCalibrationRunnerError("synthetic early write")
        return real_write(custody_value, name, body)

    monkeypatch.setattr(subject, "_write_output_record", fail_before)
    with pytest.raises(subject.SkeletonGraphCalibrationRunnerError, match="early write"):
        subject.run_calibration(
            exposure_authorization=authorization,
            preregistration_path=PREREGISTRATION,
            passed_fit=protocol,
            passed_fit_paths=_paths(),
            panel_identities=_identities(scope),
            output_directory=output,
            calibration_pixel_reader=pixels,
            inference_runner=infer,
            delayed_label_reader_factory=labels,
        )
    assert calls == []
    gap = subject.SkeletonGraphCalibrationGap.from_data(
        json.loads((output / "calibration_gap.json").read_bytes())
    )
    assert gap.stage == expected_stage
    assert not (output / "population_grant.json").exists()
    replayed = subject.cold_replay_calibration(
        run_directory=output, preregistration_path=PREREGISTRATION
    )
    assert type(replayed) is subject.SkeletonGraphCalibrationGap
    assert replayed.to_data() == gap.to_data()


@pytest.mark.parametrize(
    ("target", "expected_stage", "digest_field"),
    [
        ("raw_predictions.json", "integrity_prediction_write", "prediction_record_digest"),
        ("delayed_labels.json", "integrity_delayed_label_write", "label_record_digest"),
    ],
)
def test_postpublished_stage_failure_uses_actual_inventory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    target: str,
    expected_stage: str,
    digest_field: str,
) -> None:
    protocol = _protocol()
    _install_synthetic_metadata_boundary(monkeypatch, tmp_path, protocol)
    scope = subject.SkeletonGraphCalibrationScope.SAME_FAMILY
    output = tmp_path / ("postpublished-" + target.removesuffix(".json"))
    authorization = subject.authorize_calibration_exposure(
        scope=scope,
        preregistration_path=PREREGISTRATION,
        passed_fit=protocol,
        passed_fit_paths=_paths(),
        output_directory=output,
    )
    calls: list[str] = []
    pixels, infer, labels = _callbacks(output, calls)
    real_write = subject._write_output_record
    fired = False

    def fail_after(custody_value, name, body):
        nonlocal fired
        result = real_write(custody_value, name, body)
        if name == target and not fired:
            fired = True
            raise subject.SkeletonGraphCalibrationRunnerError(
                "synthetic failure after publication"
            )
        return result

    monkeypatch.setattr(subject, "_write_output_record", fail_after)
    with pytest.raises(subject.SkeletonGraphCalibrationRunnerError, match="after publication"):
        subject.run_calibration(
            exposure_authorization=authorization,
            preregistration_path=PREREGISTRATION,
            passed_fit=protocol,
            passed_fit_paths=_paths(),
            panel_identities=_identities(scope),
            output_directory=output,
            calibration_pixel_reader=pixels,
            inference_runner=infer,
            delayed_label_reader_factory=labels,
        )
    raw = (output / target).read_bytes()
    record = json.loads(raw)
    gap = subject.SkeletonGraphCalibrationGap.from_data(
        json.loads((output / "calibration_gap.json").read_bytes())
    )
    assert gap.stage == expected_stage
    assert getattr(gap, digest_field) == record["record_digest"]
    assert gap.integrity_custody is not None
    entry = gap.integrity_custody["inventory"][target]
    assert entry["file_sha256"] == _sha(raw)
    assert entry["record_digest"] == record["record_digest"]
    assert not (output / "population_grant.json").exists()
    assert subject.cold_replay_calibration(
        run_directory=output, preregistration_path=PREREGISTRATION
    ).to_data() == gap.to_data()


@pytest.mark.parametrize("crash_point", ["empty", "claim"])
def test_prepixel_output_root_crash_recovers_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, crash_point: str
) -> None:
    protocol = _protocol()
    _install_synthetic_metadata_boundary(monkeypatch, tmp_path, protocol)
    scope = subject.SkeletonGraphCalibrationScope.SAME_FAMILY
    output = tmp_path / f"prepixel-{crash_point}"
    authorization = subject.authorize_calibration_exposure(
        scope=scope,
        preregistration_path=PREREGISTRATION,
        passed_fit=protocol,
        passed_fit_paths=_paths(),
        output_directory=output,
    )
    calls: list[str] = []
    pixels, infer, labels = _callbacks(output, calls)
    if crash_point == "empty":
        real_acquire = subject._acquire_recoverable_output_directory

        def crash_after_mkdir(*args, **kwargs):
            custody_value, created = real_acquire(*args, **kwargs)
            custody_value.close()
            assert created is True
            raise KeyboardInterrupt("synthetic stop after mkdir")

        monkeypatch.setattr(
            subject, "_acquire_recoverable_output_directory", crash_after_mkdir
        )
    else:
        real_claim = subject._persist_or_verify_output_root_claim

        def crash_after_claim(*args, **kwargs):
            real_claim(*args, **kwargs)
            raise KeyboardInterrupt("synthetic stop after root claim")

        monkeypatch.setattr(
            subject, "_persist_or_verify_output_root_claim", crash_after_claim
        )
    with pytest.raises(KeyboardInterrupt):
        subject.run_calibration(
            exposure_authorization=authorization,
            preregistration_path=PREREGISTRATION,
            passed_fit=protocol,
            passed_fit_paths=_paths(),
            panel_identities=_identities(scope),
            output_directory=output,
            calibration_pixel_reader=pixels,
            inference_runner=infer,
            delayed_label_reader_factory=labels,
        )
    assert calls == []
    if crash_point == "empty":
        assert list(output.iterdir()) == []
        monkeypatch.setattr(
            subject, "_acquire_recoverable_output_directory", real_acquire
        )
    else:
        assert [item.name for item in output.iterdir()] == ["output_root_claim.json"]
        monkeypatch.setattr(
            subject, "_persist_or_verify_output_root_claim", real_claim
        )
    result = subject.run_calibration(
        exposure_authorization=authorization,
        preregistration_path=PREREGISTRATION,
        passed_fit=protocol,
        passed_fit_paths=_paths(),
        panel_identities=_identities(scope),
        output_directory=output,
        calibration_pixel_reader=pixels,
        inference_runner=infer,
        delayed_label_reader_factory=labels,
    )
    assert type(result) is subject.SkeletonGraphPopulationGrant
    assert calls.count("inference") == 1
    assert calls.count("label_factory") == 1
    assert calls.count("labels") == 1


def test_interrupted_partial_root_terminalizes_without_callbacks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    protocol = _protocol()
    _install_synthetic_metadata_boundary(monkeypatch, tmp_path, protocol)
    scope = subject.SkeletonGraphCalibrationScope.SAME_FAMILY
    output = tmp_path / "interrupted-partial"
    authorization = subject.authorize_calibration_exposure(
        scope=scope,
        preregistration_path=PREREGISTRATION,
        passed_fit=protocol,
        passed_fit_paths=_paths(),
        output_directory=output,
    )
    real_write = subject._write_output_record

    def stop_after_precommit(custody_value, name, body):
        result = real_write(custody_value, name, body)
        if name == "precommit.json":
            raise KeyboardInterrupt("synthetic stop after precommit")
        return result

    monkeypatch.setattr(subject, "_write_output_record", stop_after_precommit)
    with pytest.raises(KeyboardInterrupt):
        subject.run_calibration(
            exposure_authorization=authorization,
            preregistration_path=PREREGISTRATION,
            passed_fit=protocol,
            passed_fit_paths=_paths(),
            panel_identities=_identities(scope),
            output_directory=output,
            calibration_pixel_reader=lambda *_args: b"forbidden",
            inference_runner=lambda *_args: (_ for _ in ()).throw(AssertionError()),
            delayed_label_reader_factory=lambda: (_ for _ in ()).throw(AssertionError()),
        )
    monkeypatch.setattr(subject, "_write_output_record", real_write)
    opened: list[str] = []

    def forbidden(*_args, **_kwargs):
        opened.append("opened")
        raise AssertionError("callback opened during interrupted recovery")

    result = subject.run_calibration(
        exposure_authorization=authorization,
        preregistration_path=PREREGISTRATION,
        passed_fit=protocol,
        passed_fit_paths=_paths(),
        panel_identities=_identities(scope),
        output_directory=output,
        calibration_pixel_reader=forbidden,
        inference_runner=forbidden,
        delayed_label_reader_factory=forbidden,
    )
    assert type(result) is subject.SkeletonGraphCalibrationGap
    assert result.stage == "integrity_recovered_interrupted_execution"
    assert opened == []
    assert not (output / "population_grant.json").exists()
    assert subject.cold_replay_calibration(
        run_directory=output, preregistration_path=PREREGISTRATION
    ).to_data() == result.to_data()


def test_postpublished_grant_remains_the_only_terminal_winner(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    protocol = _protocol()
    _install_synthetic_metadata_boundary(monkeypatch, tmp_path, protocol)
    scope = subject.SkeletonGraphCalibrationScope.SAME_FAMILY
    output = tmp_path / "postpublished-grant"
    authorization = subject.authorize_calibration_exposure(
        scope=scope,
        preregistration_path=PREREGISTRATION,
        passed_fit=protocol,
        passed_fit_paths=_paths(),
        output_directory=output,
    )
    calls: list[str] = []
    pixels, infer, labels = _callbacks(output, calls)
    real_persist = subject._persist_terminal_outcome
    fired = False

    def fail_after_terminal(*args, **kwargs):
        nonlocal fired
        result = real_persist(*args, **kwargs)
        if type(result) is subject.SkeletonGraphPopulationGrant and not fired:
            fired = True
            raise subject.SkeletonGraphCalibrationRunnerError(
                "synthetic return-path failure after grant publication"
            )
        return result

    monkeypatch.setattr(subject, "_persist_terminal_outcome", fail_after_terminal)
    grant = subject.run_calibration(
        exposure_authorization=authorization,
        preregistration_path=PREREGISTRATION,
        passed_fit=protocol,
        passed_fit_paths=_paths(),
        panel_identities=_identities(scope),
        output_directory=output,
        calibration_pixel_reader=pixels,
        inference_runner=infer,
        delayed_label_reader_factory=labels,
    )
    assert type(grant) is subject.SkeletonGraphPopulationGrant
    assert fired is True
    assert (output / "population_grant.json").is_file()
    assert not (output / "calibration_gap.json").exists()
    replay = subject.cold_replay_calibration(
        run_directory=output, preregistration_path=PREREGISTRATION
    )
    assert type(replay) is subject.SkeletonGraphCalibrationReplayReceipt
    assert replay.verifies(grant)


@pytest.mark.parametrize(
    "mutation",
    ["stage", "reasons", "scope", "passed_fit", "inventory", "bool_size"],
)
def test_integrity_gap_cold_replay_rejects_resealed_custody_mutation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mutation: str
) -> None:
    protocol = _protocol()
    _install_synthetic_metadata_boundary(monkeypatch, tmp_path, protocol)
    scope = subject.SkeletonGraphCalibrationScope.SAME_FAMILY
    output = tmp_path / f"gap-mutation-{mutation}"
    authorization = subject.authorize_calibration_exposure(
        scope=scope,
        preregistration_path=PREREGISTRATION,
        passed_fit=protocol,
        passed_fit_paths=_paths(),
        output_directory=output,
    )
    real_write = subject._write_output_record

    def fail_precommit(custody_value, name, body):
        if name == "precommit.json":
            raise subject.SkeletonGraphCalibrationRunnerError("synthetic precommit fault")
        return real_write(custody_value, name, body)

    monkeypatch.setattr(subject, "_write_output_record", fail_precommit)
    with pytest.raises(subject.SkeletonGraphCalibrationRunnerError):
        subject.run_calibration(
            exposure_authorization=authorization,
            preregistration_path=PREREGISTRATION,
            passed_fit=protocol,
            passed_fit_paths=_paths(),
            panel_identities=_identities(scope),
            output_directory=output,
            calibration_pixel_reader=lambda *_args: b"forbidden",
            inference_runner=lambda *_args: (_ for _ in ()).throw(AssertionError()),
            delayed_label_reader_factory=lambda: (_ for _ in ()).throw(AssertionError()),
        )
    gap_path = output / "calibration_gap.json"
    original = subject.SkeletonGraphCalibrationGap.from_data(
        json.loads(gap_path.read_bytes())
    )
    assert original.stage == "integrity_precommit"
    assert original.reason_codes == (
        "execution_integrity_failure",
        "precommit_failed",
    )
    assert original.scope is authorization.scope
    assert original.passed_fit_record_digest == authorization.passed_fit_record_digest

    def mutate(value):
        if mutation == "stage":
            value["stage"] = "integrity_prediction_attempt"
            value["reason_codes"] = [
                "execution_integrity_failure",
                "prediction_attempt_failed",
            ]
            value["integrity_custody"]["failure_stage"] = "prediction_attempt"
        elif mutation == "reasons":
            value["reason_codes"] = ["execution_integrity_failure", "wrong_failed"]
        elif mutation == "scope":
            value["scope"] = subject.SkeletonGraphCalibrationScope.GENERIC_V3.value
        elif mutation == "passed_fit":
            value["passed_fit_record_digest"] = SHA_A
        elif mutation == "inventory":
            value["integrity_custody"]["inventory"]["authorization.json"][
                "file_sha256"
            ] = SHA_A
        else:
            value["integrity_custody"]["inventory"]["authorization.json"][
                "size_bytes"
            ] = True

    _reseal(gap_path, mutate)
    with pytest.raises(subject.SkeletonGraphCalibrationRunnerError):
        subject.cold_replay_calibration(
            run_directory=output, preregistration_path=PREREGISTRATION
        )


def test_unrelated_ledger_child_does_not_block_campaign_but_successor_only_does(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    protocol = _protocol()
    authority_dir = _install_synthetic_metadata_boundary(
        monkeypatch, tmp_path, protocol
    )
    registration = subject._load_preregistration(PREREGISTRATION)
    predecessor_path = ROOT / registration["exposure_predecessor"]["ledger_path"]
    predecessor = subject._decode_exposure_ledger(
        predecessor_path.read_bytes(), label="unrelated-child predecessor"
    )
    unrelated = predecessor.record(
        phase="unrelated_test_phase",
        actor="unrelated_test_actor",
        purpose="unrelated_test_purpose",
        task_ids=("unrelated_task",),
        observed_at="2026-08-10T01:02:03Z",
        require_unseen=True,
    )
    unrelated_name = unrelated.digest.removeprefix("sha256:") + ".exposure.json"
    (authority_dir / unrelated_name).write_bytes(unrelated.to_json().encode())
    output = tmp_path / "unrelated-child"
    authorization = subject.authorize_calibration_exposure(
        scope=subject.SkeletonGraphCalibrationScope.SAME_FAMILY,
        preregistration_path=PREREGISTRATION,
        passed_fit=protocol,
        passed_fit_paths=_paths(),
        output_directory=output,
    )
    assert type(authorization) is subject.SkeletonGraphCalibrationExposureAuthorization
    assert (authority_dir / unrelated_name).is_file()
    wire = authorization.to_data()
    wire["exposure_successor_filename"] = "alternate.exposure.json"
    body = dict(wire)
    body.pop("record_digest")
    wire["record_digest"] = "sha256:" + subject.canonical_digest(body)
    with pytest.raises(subject.SkeletonGraphCalibrationRunnerError):
        subject._authorization_from_data(wire)

    (authority_dir / authorization.campaign_intent_filename).unlink()
    (authority_dir / authorization.filename).unlink()
    before = sorted(path.name for path in authority_dir.glob("*.exposure.json"))
    monkeypatch.setattr(
        subject,
        "_new_observed_at",
        lambda: (_ for _ in ()).throw(AssertionError("orphan recovery rerolled")),
    )
    with pytest.raises(subject.SkeletonGraphCalibrationRunnerError, match="or successor"):
        subject.authorize_calibration_exposure(
            scope=subject.SkeletonGraphCalibrationScope.SAME_FAMILY,
            preregistration_path=PREREGISTRATION,
            passed_fit=protocol,
            passed_fit_paths=_paths(),
            output_directory=output,
        )
    assert sorted(path.name for path in authority_dir.glob("*.exposure.json")) == before


@pytest.mark.parametrize(
    "global_field",
    [
        "campaign_intent_filename",
        "exposure_predecessor_filename",
        "exposure_successor_filename",
        "filename",
    ],
)
def test_delayed_lease_freshly_verifies_every_global_chain_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, global_field: str
) -> None:
    protocol = _protocol()
    authority_dir = _install_synthetic_metadata_boundary(
        monkeypatch, tmp_path, protocol
    )
    scope = subject.SkeletonGraphCalibrationScope.SAME_FAMILY
    output = tmp_path / ("global-fresh-" + global_field)
    authorization = subject.authorize_calibration_exposure(
        scope=scope,
        preregistration_path=PREREGISTRATION,
        passed_fit=protocol,
        passed_fit_paths=_paths(),
        output_directory=output,
    )
    calls: list[str] = []
    pixels, infer, base_factory = _callbacks(output, calls)

    def factory():
        base_reader = base_factory()

        def read(request):
            filename = getattr(authorization, global_field)
            path = authority_dir / filename
            saved = path.read_bytes()
            try:
                path.write_bytes(b"tampered-global-chain\n")
                with pytest.raises(subject.SkeletonGraphCalibrationRunnerError):
                    subject.verify_and_consume_delayed_label_request(request)
                assert request._lease.consumed is False
            finally:
                path.write_bytes(saved)
            return base_reader(request)

        return read

    result = subject.run_calibration(
        exposure_authorization=authorization,
        preregistration_path=PREREGISTRATION,
        passed_fit=protocol,
        passed_fit_paths=_paths(),
        panel_identities=_identities(scope),
        output_directory=output,
        calibration_pixel_reader=pixels,
        inference_runner=infer,
        delayed_label_reader_factory=factory,
    )
    assert type(result) is subject.SkeletonGraphPopulationGrant


def test_global_chain_is_fresh_verified_immediately_before_outcome(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    protocol = _protocol()
    authority_dir = _install_synthetic_metadata_boundary(
        monkeypatch, tmp_path, protocol
    )
    scope = subject.SkeletonGraphCalibrationScope.SAME_FAMILY
    output = tmp_path / "global-before-outcome"
    authorization = subject.authorize_calibration_exposure(
        scope=scope,
        preregistration_path=PREREGISTRATION,
        passed_fit=protocol,
        passed_fit_paths=_paths(),
        output_directory=output,
    )
    calls: list[str] = []
    pixels, infer, labels = _callbacks(output, calls)
    successor_path = authority_dir / authorization.exposure_successor_filename
    saved = successor_path.read_bytes()
    real_evaluate = subject._evaluate_archive

    def tamper_after_evaluation(*args, **kwargs):
        result = real_evaluate(*args, **kwargs)
        successor_path.write_bytes(b"tampered-before-outcome\n")
        return result

    monkeypatch.setattr(subject, "_evaluate_archive", tamper_after_evaluation)
    try:
        with pytest.raises(subject.SkeletonGraphCalibrationRunnerError):
            subject.run_calibration(
                exposure_authorization=authorization,
                preregistration_path=PREREGISTRATION,
                passed_fit=protocol,
                passed_fit_paths=_paths(),
                panel_identities=_identities(scope),
                output_directory=output,
                calibration_pixel_reader=pixels,
                inference_runner=infer,
                delayed_label_reader_factory=labels,
            )
    finally:
        successor_path.write_bytes(saved)
    assert not (output / "population_grant.json").exists()
    gap = subject.SkeletonGraphCalibrationGap.from_data(
        json.loads((output / "calibration_gap.json").read_bytes())
    )
    assert gap.stage == "integrity_population_evaluation"
    assert subject.cold_replay_calibration(
        run_directory=output, preregistration_path=PREREGISTRATION
    ).to_data() == gap.to_data()


def test_cold_replay_materializes_terminal_claim_after_process_stop(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    protocol = _protocol()
    _install_synthetic_metadata_boundary(monkeypatch, tmp_path, protocol)
    scope = subject.SkeletonGraphCalibrationScope.SAME_FAMILY
    output = tmp_path / "terminal-claim-stop"
    authorization = subject.authorize_calibration_exposure(
        scope=scope,
        preregistration_path=PREREGISTRATION,
        passed_fit=protocol,
        passed_fit_paths=_paths(),
        output_directory=output,
    )
    calls: list[str] = []
    pixels, infer, labels = _callbacks(output, calls)
    real_atomic = subject._atomic_write_once_bytes
    stopped = False

    def stop_after_terminal_claim(custody_value, name, raw, **kwargs):
        nonlocal stopped
        result = real_atomic(custody_value, name, raw, **kwargs)
        if name == "terminal_state.json" and not stopped:
            stopped = True
            raise KeyboardInterrupt("synthetic stop after terminal claim")
        return result

    monkeypatch.setattr(subject, "_atomic_write_once_bytes", stop_after_terminal_claim)
    with pytest.raises(KeyboardInterrupt, match="terminal claim"):
        subject.run_calibration(
            exposure_authorization=authorization,
            preregistration_path=PREREGISTRATION,
            passed_fit=protocol,
            passed_fit_paths=_paths(),
            panel_identities=_identities(scope),
            output_directory=output,
            calibration_pixel_reader=pixels,
            inference_runner=infer,
            delayed_label_reader_factory=labels,
        )
    assert stopped is True
    assert (output / "terminal_state.json").is_file()
    assert not (output / "population_grant.json").exists()
    assert not (output / "calibration_gap.json").exists()
    calls_before_replay = tuple(calls)

    monkeypatch.setattr(subject, "_atomic_write_once_bytes", real_atomic)
    replay = subject.cold_replay_calibration(
        run_directory=output, preregistration_path=PREREGISTRATION
    )
    assert type(replay) is subject.SkeletonGraphCalibrationReplayReceipt
    assert (output / "population_grant.json").is_file()
    assert not (output / "calibration_gap.json").exists()
    assert tuple(calls) == calls_before_replay


@pytest.mark.parametrize(
    ("artifact", "field", "replacement"),
    [
        ("population_grant.json", "missing_observed_pair_probability", False),
        ("terminal_state.json", "single_terminal_winner", 1),
        ("authorization.json", "calibration_pixels_authorized", 1),
        ("precommit.json", "calibration_pixel_reads_so_far", False),
    ],
)
def test_cold_replay_rejects_numeric_type_aliases_in_exact_wires(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    artifact: str,
    field: str,
    replacement: object,
) -> None:
    protocol = _protocol()
    _install_synthetic_metadata_boundary(monkeypatch, tmp_path, protocol)
    output = tmp_path / ("typed-wire-" + artifact.removesuffix(".json"))
    _authorize_and_run_success(
        scope=subject.SkeletonGraphCalibrationScope.SAME_FAMILY,
        protocol=protocol,
        output=output,
    )
    _reseal(output / artifact, lambda value: value.__setitem__(field, replacement))
    with pytest.raises(subject.SkeletonGraphCalibrationRunnerError):
        subject.cold_replay_calibration(
            run_directory=output, preregistration_path=PREREGISTRATION
        )

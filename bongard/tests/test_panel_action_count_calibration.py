from __future__ import annotations

import copy

import pytest

from bongard.evidence import Disposition
from bongard.panel_action_count_calibration import (
    ActionCountAxis,
    ActionCountCalibrationArtifact,
    ActionCountCalibrationError,
    ActionCountCalibrationInput,
    ActionCountCalibrationStatus,
    LabeledActionCountCalibrationPanel,
    LabeledActionCountCalibrationTask,
    RawActionCountObservation,
    apply_action_count_calibration,
    cold_replay_action_count_calibration,
    verify_calibrated_action_count_observation,
)


def _address(char: str) -> str:
    return "sha256:" + char * 64


def _input(
    *,
    straight_outlier: int | None = None,
    arc_outlier: int | None = None,
    error_at: tuple[int, int] | None = None,
) -> ActionCountCalibrationInput:
    tasks = []
    for task_index in range(20):
        panels = []
        for panel_index in range(14):
            is_last = task_index == 19 and panel_index == 13
            straight_truth = straight_outlier if is_last and straight_outlier is not None else 4
            arc_truth = arc_outlier if is_last and arc_outlier is not None else 2
            observation = RawActionCountObservation(
                0 if is_last and straight_outlier is not None else straight_truth,
                0 if is_last and straight_outlier is not None else straight_truth,
                0 if is_last and arc_outlier is not None else arc_truth,
                0 if is_last and arc_outlier is not None else arc_truth,
                "unreadable" if error_at == (task_index, panel_index) else None,
            )
            panels.append(
                LabeledActionCountCalibrationPanel(
                    f"task_{task_index:02d}/panel_{panel_index:02d}",
                    observation,
                    straight_truth,
                    arc_truth,
                )
            )
        tasks.append(
            LabeledActionCountCalibrationTask.create(
                f"task_{task_index:02d}", panels
            )
        )
    return ActionCountCalibrationInput.freeze(
        plan_record_digest=_address("a"),
        prediction_batch_digest=_address("b"),
        label_release_digest=_address("c"),
        measurement_result_digest=_address("d"),
        observer_protocol_digest=_address("e"),
        tasks=tasks,
    )


def test_task_max_zero_omission_grant_is_conservative_and_cold_replays() -> None:
    artifact = ActionCountCalibrationArtifact.derive(
        _input(straight_outlier=8, arc_outlier=3)
    )

    assert artifact.status is ActionCountCalibrationStatus.GRANTED
    assert artifact.straight_radius == 8
    assert artifact.arc_radius == 3
    assert sum(artifact.straight_panel_residual_histogram) == 280
    assert sum(artifact.arc_panel_residual_histogram) == 280
    assert max(artifact.straight_task_max_residuals) == 8
    assert max(artifact.arc_task_max_residuals) == 3
    assert artifact.to_data()["model_calls_for_derivation_or_replay"] == 0

    restored = cold_replay_action_count_calibration(
        artifact, expected_artifact_address=artifact.artifact_address
    )
    assert restored == artifact

    calibrated = apply_action_count_calibration(
        artifact, RawActionCountObservation(4, 4, 2, 2)
    )
    assert (
        verify_calibrated_action_count_observation(
            calibrated,
            artifact=artifact,
            raw_observation=RawActionCountObservation(4, 4, 2, 2),
        )
        == calibrated
    )
    assert calibrated.interval(ActionCountAxis.STRAIGHT) == (0, 9)
    assert calibrated.interval(ActionCountAxis.ARC) == (0, 5)
    assert (
        calibrated.equality_disposition(ActionCountAxis.STRAIGHT, 4)
        is Disposition.INDETERMINATE
    )
    assert (
        calibrated.equality_disposition(ActionCountAxis.ARC, 8)
        is Disposition.CERTIFIED_ABSENT
    )


def test_zero_radius_grant_produces_present_absent_indeterminate_and_error() -> None:
    artifact = ActionCountCalibrationArtifact.derive(_input())
    assert artifact.straight_radius == artifact.arc_radius == 0

    exact = apply_action_count_calibration(
        artifact, RawActionCountObservation(4, 4, 2, 2)
    )
    assert (
        exact.equality_disposition(ActionCountAxis.STRAIGHT, 4)
        is Disposition.PRESENT
    )
    assert (
        exact.equality_disposition(ActionCountAxis.STRAIGHT, 5)
        is Disposition.CERTIFIED_ABSENT
    )

    wide = apply_action_count_calibration(
        artifact, RawActionCountObservation(3, 5, 1, 3)
    )
    assert (
        wide.equality_disposition(ActionCountAxis.STRAIGHT, 4)
        is Disposition.INDETERMINATE
    )

    failed = apply_action_count_calibration(
        artifact, RawActionCountObservation(0, 9, 0, 9, "unreadable")
    )
    assert (
        failed.equality_disposition(ActionCountAxis.STRAIGHT, 4)
        is Disposition.ERROR
    )


def test_any_calibration_error_emits_gap_and_cannot_project() -> None:
    artifact = ActionCountCalibrationArtifact.derive(_input(error_at=(3, 7)))
    assert artifact.status is ActionCountCalibrationStatus.GAP
    assert artifact.straight_radius is None
    assert artifact.arc_radius is None
    assert artifact.error_panel_keys == ("task_03/panel_07",)
    with pytest.raises(ActionCountCalibrationError):
        apply_action_count_calibration(
            artifact, RawActionCountObservation(4, 4, 2, 2)
        )


def test_round_trip_and_tamper_checks_bind_exact_policy_and_inputs() -> None:
    artifact = ActionCountCalibrationArtifact.derive(_input())
    assert ActionCountCalibrationArtifact.from_data(artifact.to_data()) == artifact

    changed = copy.deepcopy(artifact.to_data())
    changed["straight_radius"] = 1
    with pytest.raises(ActionCountCalibrationError):
        ActionCountCalibrationArtifact.from_data(changed)

    changed = copy.deepcopy(artifact.to_data())
    changed["stratified_or_target_count_radius_selection"] = True
    with pytest.raises(ActionCountCalibrationError):
        ActionCountCalibrationArtifact.from_data(changed)

    changed = copy.deepcopy(artifact.to_data())
    changed["calibration_input"]["phase"] = "fit"
    with pytest.raises(ActionCountCalibrationError):
        ActionCountCalibrationArtifact.from_data(changed)

    with pytest.raises(ActionCountCalibrationError):
        cold_replay_action_count_calibration(
            artifact, expected_artifact_address=_address("f")
        )

    raw = RawActionCountObservation(4, 4, 2, 2)
    calibrated = apply_action_count_calibration(artifact, raw)
    changed = copy.deepcopy(calibrated.to_data())
    changed["straight_action_count_upper"] = 5
    with pytest.raises(ActionCountCalibrationError):
        type(calibrated).from_data(changed)

    forged = type(calibrated)(
        calibrated.calibration_artifact_address,
        calibrated.raw_observation_digest,
        0,
        9,
        calibrated.arc_lower,
        calibrated.arc_upper,
        calibrated.error_code,
    )
    with pytest.raises(ActionCountCalibrationError):
        verify_calibrated_action_count_observation(
            forged, artifact=artifact, raw_observation=raw
        )

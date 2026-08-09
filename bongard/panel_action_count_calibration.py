"""Task-level calibration authority for typed action-count observations.

The vision model is allowed to emit only raw inclusive integer intervals for
straight and arc carrier-action counts.  This module never calls a model.  It
derives one expansion radius per axis from an independently frozen calibration
cohort by taking the maximum residual over every panel of every task.  Errors
fail calibration closed.  The resulting grant is then the sole authority for
mapping future raw intervals to the four dispositions used by Python formulas.

The deliberately conservative task-maximum rule protects a zero-contradiction
support gate.  It may make an observer non-decisive; calibration is not allowed
to manufacture information that vision did not supply.  Lean is absent and
removable.
"""

from __future__ import annotations

from bongard.runtime_source_snapshot import capture_loaded_source, verify_loaded_source


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)

from dataclasses import dataclass
from enum import Enum
import re
from typing import Any, Mapping, Sequence

from bongard.canonical import canonical_digest
from bongard.evidence import Disposition


CALIBRATION_TASK_COUNT = 20
PANELS_PER_TASK = 14
COUNT_MINIMUM = 0
COUNT_MAXIMUM = 9

RAW_OBSERVATION_SCHEMA = "gkm.bongard-raw-action-count-observation.v1"
CALIBRATION_PANEL_SCHEMA = "gkm.bongard-action-count-calibration-panel.v1"
CALIBRATION_TASK_SCHEMA = "gkm.bongard-action-count-calibration-task.v1"
CALIBRATION_INPUT_SCHEMA = "gkm.bongard-action-count-calibration-input.v1"
CALIBRATION_ARTIFACT_SCHEMA = "gkm.bongard-action-count-calibration-artifact.v1"
CALIBRATED_OBSERVATION_SCHEMA = "gkm.bongard-calibrated-action-count-observation.v1"
CALIBRATION_ALGORITHM_ID = (
    "bongard.action-count-calibration/task-max-zero-omission-radius-v1"
)

_ADDRESS = re.compile(r"sha256:[0-9a-f]{64}\Z")
_KEY = re.compile(r"[A-Za-z0-9][A-Za-z0-9_./:-]{0,511}\Z")
_ERROR = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")


class ActionCountCalibrationError(ValueError):
    """A calibration input, grant, observation, or replay differs."""


class ActionCountAxis(str, Enum):
    STRAIGHT = "straight"
    ARC = "arc"


class ActionCountCalibrationStatus(str, Enum):
    GRANTED = "granted"
    GAP = "gap"


def panel_action_count_calibration_source_digest() -> str:
    return verify_loaded_source(__name__, expected_source_sha256=_LOADED_SOURCE_SHA256)


def _fields(value: object, expected: set[str], label: str) -> Mapping[str, Any]:
    if (
        not isinstance(value, Mapping)
        or any(type(key) is not str for key in value)
        or set(value) != expected
    ):
        raise ActionCountCalibrationError(f"{label} fields differ")
    return value


def _address(value: object, label: str) -> str:
    if type(value) is not str or _ADDRESS.fullmatch(value) is None:
        raise ActionCountCalibrationError(f"{label} must be a sha256: address")
    return value


def _key(value: object, label: str) -> str:
    if type(value) is not str or _KEY.fullmatch(value) is None:
        raise ActionCountCalibrationError(f"{label} is not a bounded key")
    return value


def _bound(value: object, label: str) -> int:
    if type(value) is not int or not COUNT_MINIMUM <= value <= COUNT_MAXIMUM:
        raise ActionCountCalibrationError(f"{label} lies outside 0..9")
    return value


def _residual(truth: int, lower: int, upper: int) -> int:
    return max(lower - truth, truth - upper, 0)


@dataclass(frozen=True, slots=True)
class RawActionCountObservation:
    straight_lower: int
    straight_upper: int
    arc_lower: int
    arc_upper: int
    error_code: str | None = None

    def __post_init__(self) -> None:
        values = (
            _bound(self.straight_lower, "straight lower"),
            _bound(self.straight_upper, "straight upper"),
            _bound(self.arc_lower, "arc lower"),
            _bound(self.arc_upper, "arc upper"),
        )
        if values[0] > values[1] or values[2] > values[3]:
            raise ActionCountCalibrationError("raw action-count interval is reversed")
        if self.error_code is not None and (
            type(self.error_code) is not str
            or _ERROR.fullmatch(self.error_code) is None
        ):
            raise ActionCountCalibrationError("raw observation error code differs")

    @property
    def observation_digest(self) -> str:
        return canonical_digest(self.to_data())

    def interval(self, axis: ActionCountAxis) -> tuple[int, int]:
        if type(axis) is not ActionCountAxis:
            raise TypeError("axis must be exact ActionCountAxis")
        if axis is ActionCountAxis.STRAIGHT:
            return self.straight_lower, self.straight_upper
        return self.arc_lower, self.arc_upper

    def to_data(self) -> dict[str, object]:
        return {
            "schema": RAW_OBSERVATION_SCHEMA,
            "straight_action_count_lower": self.straight_lower,
            "straight_action_count_upper": self.straight_upper,
            "arc_action_count_lower": self.arc_lower,
            "arc_action_count_upper": self.arc_upper,
            "error_code": self.error_code,
        }

    @classmethod
    def from_data(cls, value: object) -> "RawActionCountObservation":
        raw = _fields(
            value,
            {
                "schema",
                "straight_action_count_lower",
                "straight_action_count_upper",
                "arc_action_count_lower",
                "arc_action_count_upper",
                "error_code",
            },
            "raw action-count observation",
        )
        if raw["schema"] != RAW_OBSERVATION_SCHEMA:
            raise ActionCountCalibrationError("raw observation schema differs")
        result = cls(
            raw["straight_action_count_lower"],
            raw["straight_action_count_upper"],
            raw["arc_action_count_lower"],
            raw["arc_action_count_upper"],
            raw["error_code"],
        )
        if result.to_data() != dict(raw):
            raise ActionCountCalibrationError("raw observation is not canonical")
        return result


@dataclass(frozen=True, slots=True)
class LabeledActionCountCalibrationPanel:
    panel_key: str
    observation: RawActionCountObservation
    true_straight_count: int
    true_arc_count: int

    def __post_init__(self) -> None:
        _key(self.panel_key, "calibration panel key")
        if type(self.observation) is not RawActionCountObservation:
            raise TypeError("calibration panel needs exact raw observation")
        _bound(self.true_straight_count, "true straight count")
        _bound(self.true_arc_count, "true arc count")

    def residual(self, axis: ActionCountAxis) -> int:
        lower, upper = self.observation.interval(axis)
        truth = (
            self.true_straight_count
            if axis is ActionCountAxis.STRAIGHT
            else self.true_arc_count
        )
        return _residual(truth, lower, upper)

    def to_data(self) -> dict[str, object]:
        return {
            "schema": CALIBRATION_PANEL_SCHEMA,
            "panel_key": self.panel_key,
            "observation": self.observation.to_data(),
            "true_straight_action_count": self.true_straight_count,
            "true_arc_action_count": self.true_arc_count,
        }

    @classmethod
    def from_data(cls, value: object) -> "LabeledActionCountCalibrationPanel":
        raw = _fields(
            value,
            {
                "schema",
                "panel_key",
                "observation",
                "true_straight_action_count",
                "true_arc_action_count",
            },
            "labeled action-count panel",
        )
        if raw["schema"] != CALIBRATION_PANEL_SCHEMA:
            raise ActionCountCalibrationError("calibration panel schema differs")
        result = cls(
            raw["panel_key"],
            RawActionCountObservation.from_data(raw["observation"]),
            raw["true_straight_action_count"],
            raw["true_arc_action_count"],
        )
        if result.to_data() != dict(raw):
            raise ActionCountCalibrationError("calibration panel is not canonical")
        return result


@dataclass(frozen=True, slots=True)
class LabeledActionCountCalibrationTask:
    task_key: str
    panels: tuple[LabeledActionCountCalibrationPanel, ...]

    def __post_init__(self) -> None:
        _key(self.task_key, "calibration task key")
        if (
            type(self.panels) is not tuple
            or len(self.panels) != PANELS_PER_TASK
            or any(type(item) is not LabeledActionCountCalibrationPanel for item in self.panels)
            or len({item.panel_key for item in self.panels}) != PANELS_PER_TASK
            or tuple(sorted(self.panels, key=lambda item: item.panel_key)) != self.panels
        ):
            raise ActionCountCalibrationError(
                "calibration task needs fourteen unique key-sorted panels"
            )

    @classmethod
    def create(
        cls,
        task_key: str,
        panels: Sequence[LabeledActionCountCalibrationPanel],
    ) -> "LabeledActionCountCalibrationTask":
        return cls(task_key, tuple(sorted(panels, key=lambda item: item.panel_key)))

    def task_max_residual(self, axis: ActionCountAxis) -> int:
        return max(item.residual(axis) for item in self.panels)

    def to_data(self) -> dict[str, object]:
        return {
            "schema": CALIBRATION_TASK_SCHEMA,
            "task_key": self.task_key,
            "panels": [item.to_data() for item in self.panels],
            "panel_count": PANELS_PER_TASK,
        }

    @classmethod
    def from_data(cls, value: object) -> "LabeledActionCountCalibrationTask":
        raw = _fields(
            value,
            {"schema", "task_key", "panels", "panel_count"},
            "labeled action-count task",
        )
        if (
            raw["schema"] != CALIBRATION_TASK_SCHEMA
            or raw["panel_count"] != PANELS_PER_TASK
            or type(raw["panels"]) is not list
        ):
            raise ActionCountCalibrationError("calibration task policy differs")
        result = cls(
            raw["task_key"],
            tuple(LabeledActionCountCalibrationPanel.from_data(item) for item in raw["panels"]),
        )
        if result.to_data() != dict(raw):
            raise ActionCountCalibrationError("calibration task is not canonical")
        return result


@dataclass(frozen=True, slots=True)
class ActionCountCalibrationInput:
    plan_record_digest: str
    prediction_batch_digest: str
    label_release_digest: str
    measurement_result_digest: str
    observer_protocol_digest: str
    tasks: tuple[LabeledActionCountCalibrationTask, ...]

    def __post_init__(self) -> None:
        for label, value in (
            ("plan record", self.plan_record_digest),
            ("prediction batch", self.prediction_batch_digest),
            ("label release", self.label_release_digest),
            ("measurement result", self.measurement_result_digest),
            ("observer protocol", self.observer_protocol_digest),
        ):
            _address(value, label + " digest")
        if (
            type(self.tasks) is not tuple
            or len(self.tasks) != CALIBRATION_TASK_COUNT
            or any(type(item) is not LabeledActionCountCalibrationTask for item in self.tasks)
            or len({item.task_key for item in self.tasks}) != CALIBRATION_TASK_COUNT
            or tuple(sorted(self.tasks, key=lambda item: item.task_key)) != self.tasks
        ):
            raise ActionCountCalibrationError(
                "calibration input needs twenty unique key-sorted tasks"
            )

    @classmethod
    def freeze(
        cls,
        *,
        plan_record_digest: str,
        prediction_batch_digest: str,
        label_release_digest: str,
        measurement_result_digest: str,
        observer_protocol_digest: str,
        tasks: Sequence[LabeledActionCountCalibrationTask],
    ) -> "ActionCountCalibrationInput":
        return cls(
            plan_record_digest,
            prediction_batch_digest,
            label_release_digest,
            measurement_result_digest,
            observer_protocol_digest,
            tuple(sorted(tasks, key=lambda item: item.task_key)),
        )

    @property
    def input_digest(self) -> str:
        return canonical_digest(self.content_data())

    def content_data(self) -> dict[str, object]:
        return {
            "schema": CALIBRATION_INPUT_SCHEMA,
            "phase": "calibration",
            "plan_record_digest": self.plan_record_digest,
            "prediction_batch_digest": self.prediction_batch_digest,
            "label_release_digest": self.label_release_digest,
            "measurement_result_digest": self.measurement_result_digest,
            "observer_protocol_digest": self.observer_protocol_digest,
            "tasks": [item.to_data() for item in self.tasks],
            "task_count": CALIBRATION_TASK_COUNT,
            "panels_per_task": PANELS_PER_TASK,
            "labels_opened_after_prediction_batch_fsync": True,
            "task_side_family_count_or_decoration_stratification": False,
            "target_semantic_closure_excluded": True,
            "python_is_canonical_authority": True,
            "lean_present": False,
            "lean_required": False,
        }

    def to_data(self) -> dict[str, object]:
        return {**self.content_data(), "input_digest": self.input_digest}

    @classmethod
    def from_data(cls, value: object) -> "ActionCountCalibrationInput":
        raw = _fields(
            value,
            {
                "schema",
                "phase",
                "plan_record_digest",
                "prediction_batch_digest",
                "label_release_digest",
                "measurement_result_digest",
                "observer_protocol_digest",
                "tasks",
                "task_count",
                "panels_per_task",
                "labels_opened_after_prediction_batch_fsync",
                "task_side_family_count_or_decoration_stratification",
                "target_semantic_closure_excluded",
                "python_is_canonical_authority",
                "lean_present",
                "lean_required",
                "input_digest",
            },
            "action-count calibration input",
        )
        if (
            raw["schema"] != CALIBRATION_INPUT_SCHEMA
            or raw["phase"] != "calibration"
            or raw["task_count"] != CALIBRATION_TASK_COUNT
            or raw["panels_per_task"] != PANELS_PER_TASK
            or raw["labels_opened_after_prediction_batch_fsync"] is not True
            or raw["task_side_family_count_or_decoration_stratification"] is not False
            or raw["target_semantic_closure_excluded"] is not True
            or raw["python_is_canonical_authority"] is not True
            or raw["lean_present"] is not False
            or raw["lean_required"] is not False
            or type(raw["tasks"]) is not list
        ):
            raise ActionCountCalibrationError("calibration input policy differs")
        result = cls(
            raw["plan_record_digest"],
            raw["prediction_batch_digest"],
            raw["label_release_digest"],
            raw["measurement_result_digest"],
            raw["observer_protocol_digest"],
            tuple(LabeledActionCountCalibrationTask.from_data(item) for item in raw["tasks"]),
        )
        if raw["input_digest"] != result.input_digest or result.to_data() != dict(raw):
            raise ActionCountCalibrationError("calibration input is not canonical")
        return result


def _residual_histogram(
    value: ActionCountCalibrationInput,
    axis: ActionCountAxis,
) -> tuple[int, ...]:
    counts = [0] * (COUNT_MAXIMUM + 1)
    for task in value.tasks:
        for panel in task.panels:
            counts[panel.residual(axis)] += 1
    return tuple(counts)


@dataclass(frozen=True, slots=True)
class ActionCountCalibrationArtifact:
    calibration_input: ActionCountCalibrationInput
    status: ActionCountCalibrationStatus
    straight_radius: int | None
    arc_radius: int | None
    error_panel_keys: tuple[str, ...]
    straight_panel_residual_histogram: tuple[int, ...]
    arc_panel_residual_histogram: tuple[int, ...]
    straight_task_max_residuals: tuple[int, ...]
    arc_task_max_residuals: tuple[int, ...]

    def __post_init__(self) -> None:
        if type(self.calibration_input) is not ActionCountCalibrationInput:
            raise TypeError("calibration artifact needs exact input")
        if type(self.status) is not ActionCountCalibrationStatus:
            raise TypeError("calibration status differs")
        expected_errors = tuple(
            sorted(
                panel.panel_key
                for task in self.calibration_input.tasks
                for panel in task.panels
                if panel.observation.error_code is not None
            )
        )
        expected_straight_histogram = _residual_histogram(
            self.calibration_input, ActionCountAxis.STRAIGHT
        )
        expected_arc_histogram = _residual_histogram(
            self.calibration_input, ActionCountAxis.ARC
        )
        expected_straight_maxima = tuple(
            task.task_max_residual(ActionCountAxis.STRAIGHT)
            for task in self.calibration_input.tasks
        )
        expected_arc_maxima = tuple(
            task.task_max_residual(ActionCountAxis.ARC)
            for task in self.calibration_input.tasks
        )
        expected_status = (
            ActionCountCalibrationStatus.GAP
            if expected_errors
            else ActionCountCalibrationStatus.GRANTED
        )
        expected_straight_radius = (
            None if expected_errors else max(expected_straight_maxima)
        )
        expected_arc_radius = None if expected_errors else max(expected_arc_maxima)
        if (
            self.error_panel_keys != expected_errors
            or self.status is not expected_status
            or self.straight_radius != expected_straight_radius
            or self.arc_radius != expected_arc_radius
            or self.straight_panel_residual_histogram != expected_straight_histogram
            or self.arc_panel_residual_histogram != expected_arc_histogram
            or self.straight_task_max_residuals != expected_straight_maxima
            or self.arc_task_max_residuals != expected_arc_maxima
        ):
            raise ActionCountCalibrationError("calibration artifact derivation differs")

    @classmethod
    def derive(
        cls,
        calibration_input: ActionCountCalibrationInput,
    ) -> "ActionCountCalibrationArtifact":
        if type(calibration_input) is not ActionCountCalibrationInput:
            raise TypeError("calibration derivation needs exact input")
        errors = tuple(
            sorted(
                panel.panel_key
                for task in calibration_input.tasks
                for panel in task.panels
                if panel.observation.error_code is not None
            )
        )
        straight_maxima = tuple(
            task.task_max_residual(ActionCountAxis.STRAIGHT)
            for task in calibration_input.tasks
        )
        arc_maxima = tuple(
            task.task_max_residual(ActionCountAxis.ARC)
            for task in calibration_input.tasks
        )
        return cls(
            calibration_input,
            ActionCountCalibrationStatus.GAP
            if errors
            else ActionCountCalibrationStatus.GRANTED,
            None if errors else max(straight_maxima),
            None if errors else max(arc_maxima),
            errors,
            _residual_histogram(calibration_input, ActionCountAxis.STRAIGHT),
            _residual_histogram(calibration_input, ActionCountAxis.ARC),
            straight_maxima,
            arc_maxima,
        )

    @property
    def artifact_digest(self) -> str:
        return canonical_digest(self.content_data())

    @property
    def artifact_address(self) -> str:
        return "sha256:" + self.artifact_digest

    @property
    def grant_available(self) -> bool:
        return self.status is ActionCountCalibrationStatus.GRANTED

    def radius(self, axis: ActionCountAxis) -> int:
        if not self.grant_available:
            raise ActionCountCalibrationError("calibration gap has no radius grant")
        if type(axis) is not ActionCountAxis:
            raise TypeError("axis must be exact ActionCountAxis")
        result = self.straight_radius if axis is ActionCountAxis.STRAIGHT else self.arc_radius
        assert result is not None
        return result

    def content_data(self) -> dict[str, object]:
        return {
            "schema": CALIBRATION_ARTIFACT_SCHEMA,
            "algorithm_id": CALIBRATION_ALGORITHM_ID,
            "algorithm_source_digest": panel_action_count_calibration_source_digest(),
            "calibration_input": self.calibration_input.to_data(),
            "calibration_input_digest": self.calibration_input.input_digest,
            "status": self.status.value,
            "straight_radius": self.straight_radius,
            "arc_radius": self.arc_radius,
            "error_panel_keys": list(self.error_panel_keys),
            "straight_panel_residual_histogram": list(
                self.straight_panel_residual_histogram
            ),
            "arc_panel_residual_histogram": list(self.arc_panel_residual_histogram),
            "straight_task_max_residuals": list(self.straight_task_max_residuals),
            "arc_task_max_residuals": list(self.arc_task_max_residuals),
            "radius_rule": "maximum_residual_over_every_panel_of_every_calibration_task",
            "zero_calibration_omission_required": True,
            "error_rows_fail_calibration_closed": True,
            "stratified_or_target_count_radius_selection": False,
            "heldout_threshold_adjustment_allowed": False,
            "model_calls_for_derivation_or_replay": 0,
            "python_is_canonical_authority": True,
            "lean_present": False,
            "lean_required": False,
        }

    def to_data(self) -> dict[str, object]:
        return {**self.content_data(), "artifact_digest": self.artifact_digest}

    @classmethod
    def from_data(cls, value: object) -> "ActionCountCalibrationArtifact":
        raw = _fields(
            value,
            {
                "schema",
                "algorithm_id",
                "algorithm_source_digest",
                "calibration_input",
                "calibration_input_digest",
                "status",
                "straight_radius",
                "arc_radius",
                "error_panel_keys",
                "straight_panel_residual_histogram",
                "arc_panel_residual_histogram",
                "straight_task_max_residuals",
                "arc_task_max_residuals",
                "radius_rule",
                "zero_calibration_omission_required",
                "error_rows_fail_calibration_closed",
                "stratified_or_target_count_radius_selection",
                "heldout_threshold_adjustment_allowed",
                "model_calls_for_derivation_or_replay",
                "python_is_canonical_authority",
                "lean_present",
                "lean_required",
                "artifact_digest",
            },
            "action-count calibration artifact",
        )
        if (
            raw["schema"] != CALIBRATION_ARTIFACT_SCHEMA
            or raw["algorithm_id"] != CALIBRATION_ALGORITHM_ID
            or raw["algorithm_source_digest"]
            != panel_action_count_calibration_source_digest()
            or raw["radius_rule"]
            != "maximum_residual_over_every_panel_of_every_calibration_task"
            or raw["zero_calibration_omission_required"] is not True
            or raw["error_rows_fail_calibration_closed"] is not True
            or raw["stratified_or_target_count_radius_selection"] is not False
            or raw["heldout_threshold_adjustment_allowed"] is not False
            or raw["model_calls_for_derivation_or_replay"] != 0
            or raw["python_is_canonical_authority"] is not True
            or raw["lean_present"] is not False
            or raw["lean_required"] is not False
            or type(raw["error_panel_keys"]) is not list
            or type(raw["straight_panel_residual_histogram"]) is not list
            or type(raw["arc_panel_residual_histogram"]) is not list
            or type(raw["straight_task_max_residuals"]) is not list
            or type(raw["arc_task_max_residuals"]) is not list
        ):
            raise ActionCountCalibrationError("calibration artifact policy differs")
        try:
            status = ActionCountCalibrationStatus(raw["status"])
        except Exception as exc:
            raise ActionCountCalibrationError("calibration artifact status differs") from exc
        result = cls(
            ActionCountCalibrationInput.from_data(raw["calibration_input"]),
            status,
            raw["straight_radius"],
            raw["arc_radius"],
            tuple(raw["error_panel_keys"]),
            tuple(raw["straight_panel_residual_histogram"]),
            tuple(raw["arc_panel_residual_histogram"]),
            tuple(raw["straight_task_max_residuals"]),
            tuple(raw["arc_task_max_residuals"]),
        )
        if (
            raw["calibration_input_digest"] != result.calibration_input.input_digest
            or raw["artifact_digest"] != result.artifact_digest
            or result.to_data() != dict(raw)
        ):
            raise ActionCountCalibrationError("calibration artifact is not canonical")
        return result


@dataclass(frozen=True, slots=True)
class CalibratedActionCountObservation:
    calibration_artifact_address: str
    raw_observation_digest: str
    straight_lower: int
    straight_upper: int
    arc_lower: int
    arc_upper: int
    error_code: str | None

    def __post_init__(self) -> None:
        _address(self.calibration_artifact_address, "calibration artifact")
        if type(self.raw_observation_digest) is not str or not re.fullmatch(
            r"[0-9a-f]{64}", self.raw_observation_digest
        ):
            raise ActionCountCalibrationError("raw observation digest differs")
        _bound(self.straight_lower, "calibrated straight lower")
        _bound(self.straight_upper, "calibrated straight upper")
        _bound(self.arc_lower, "calibrated arc lower")
        _bound(self.arc_upper, "calibrated arc upper")
        if self.straight_lower > self.straight_upper or self.arc_lower > self.arc_upper:
            raise ActionCountCalibrationError("calibrated interval is reversed")
        if self.error_code is not None and (
            type(self.error_code) is not str or _ERROR.fullmatch(self.error_code) is None
        ):
            raise ActionCountCalibrationError("calibrated error code differs")

    def interval(self, axis: ActionCountAxis) -> tuple[int, int]:
        if type(axis) is not ActionCountAxis:
            raise TypeError("axis must be exact ActionCountAxis")
        if axis is ActionCountAxis.STRAIGHT:
            return self.straight_lower, self.straight_upper
        return self.arc_lower, self.arc_upper

    def equality_disposition(self, axis: ActionCountAxis, value: int) -> Disposition:
        target = _bound(value, "action-count equality target")
        if self.error_code is not None:
            return Disposition.ERROR
        lower, upper = self.interval(axis)
        if lower == upper == target:
            return Disposition.PRESENT
        if target < lower or target > upper:
            return Disposition.CERTIFIED_ABSENT
        return Disposition.INDETERMINATE

    def to_data(self) -> dict[str, object]:
        return {
            "schema": CALIBRATED_OBSERVATION_SCHEMA,
            "calibration_artifact_address": self.calibration_artifact_address,
            "raw_observation_digest": self.raw_observation_digest,
            "straight_action_count_lower": self.straight_lower,
            "straight_action_count_upper": self.straight_upper,
            "arc_action_count_lower": self.arc_lower,
            "arc_action_count_upper": self.arc_upper,
            "error_code": self.error_code,
            "failed_fit_counts_as_absence": False,
        }


def apply_action_count_calibration(
    artifact: ActionCountCalibrationArtifact,
    observation: RawActionCountObservation,
) -> CalibratedActionCountObservation:
    """Expand a raw observation under an exact granted calibration artifact."""

    if type(artifact) is not ActionCountCalibrationArtifact:
        raise TypeError("calibration application needs exact artifact")
    if type(observation) is not RawActionCountObservation:
        raise TypeError("calibration application needs exact raw observation")
    if not artifact.grant_available:
        raise ActionCountCalibrationError("calibration gap cannot project observations")
    straight_radius = artifact.radius(ActionCountAxis.STRAIGHT)
    arc_radius = artifact.radius(ActionCountAxis.ARC)
    return CalibratedActionCountObservation(
        artifact.artifact_address,
        observation.observation_digest,
        max(COUNT_MINIMUM, observation.straight_lower - straight_radius),
        min(COUNT_MAXIMUM, observation.straight_upper + straight_radius),
        max(COUNT_MINIMUM, observation.arc_lower - arc_radius),
        min(COUNT_MAXIMUM, observation.arc_upper + arc_radius),
        observation.error_code,
    )


def cold_replay_action_count_calibration(
    artifact: ActionCountCalibrationArtifact,
    *,
    expected_artifact_address: str | None = None,
) -> ActionCountCalibrationArtifact:
    """Canonical zero-call replay of the complete calibration derivation."""

    if type(artifact) is not ActionCountCalibrationArtifact:
        raise TypeError("calibration replay needs exact artifact")
    restored = ActionCountCalibrationArtifact.from_data(artifact.to_data())
    if expected_artifact_address is not None and restored.artifact_address != _address(
        expected_artifact_address, "expected calibration artifact"
    ):
        raise ActionCountCalibrationError("calibration artifact address differs")
    return restored


__all__ = (
    "ActionCountAxis",
    "ActionCountCalibrationArtifact",
    "ActionCountCalibrationError",
    "ActionCountCalibrationInput",
    "ActionCountCalibrationStatus",
    "CalibratedActionCountObservation",
    "LabeledActionCountCalibrationPanel",
    "LabeledActionCountCalibrationTask",
    "RawActionCountObservation",
    "apply_action_count_calibration",
    "cold_replay_action_count_calibration",
    "panel_action_count_calibration_source_digest",
)

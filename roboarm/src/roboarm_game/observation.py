"""Public C920s image and RoArm controller-feedback observation contract.

The operational round exposes two synchronized sensor products:

* one processed RGB8 frame from a simulated Logitech C920s capture; and
* one structured packet mirroring Waveshare's T=105/T=1051 exchange.

Neither product contains private object identities, target predicates, safety
verdicts, or direct access to the simulator state.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import math
from collections.abc import Mapping
from math import pi
from typing import Final

import numpy as np
from numpy.typing import NDArray

from .interface import (
    AZIMUTH_STEP_DEG,
    HEIGHT_STEP_M,
    REACH_STEP_M,
)
from .config import BASE_AXIS_X_OFFSET_M
from .hardware import (
    C920S_DIAGONAL_FOV_DEG,
    C920S_FRAME_PERIOD_S,
    C920S_MODEL,
    C920S_SOURCE_FORMAT,
    C920S_SOURCE_FPS,
    C920S_SOURCE_SHAPE,
    C920S_TRANSPORT,
    ROARM_FEEDBACK_COMMAND,
    ROARM_MODEL,
    ROARM_SERIAL_BAUD,
    gripper_angle_from_aperture,
    roarm_feedback,
)
from .state import CalibrationState
from .world_state import WorldState

RgbFrame = NDArray[np.uint8]

OBSERVATION_SCHEMA_VERSION: Final[int] = 3
SENSOR_CONTRACT_ID: Final[str] = "rb01-roarm-c920-v3"
FRAME_ENCODING: Final[str] = "rgb8"
FRAME_SHAPE: Final[tuple[int, int, int]] = (72, 128, 3)

CAMERA_MODEL: Final[dict[str, object]] = {
    "device": C920S_MODEL,
    "transport": C920S_TRANSPORT,
    "source_format": C920S_SOURCE_FORMAT,
    "source_shape": list(C920S_SOURCE_SHAPE),
    "source_fps": C920S_SOURCE_FPS,
    "diagonal_fov_deg": C920S_DIAGONAL_FOV_DEG,
    "projection_model": "pinhole_approximation",
    "encoding": FRAME_ENCODING,
    "shape": list(FRAME_SHAPE),
    "vertical_fov_deg": 43.3,
    "position_m": [0.72, -0.10, 0.50],
    "target_m": [0.14, 0.07, 0.13],
    "up_axis": [0.0, 0.0, 1.0],
    "near_m": 0.01,
    "far_m": 5.0,
    "distortion_calibration": "unavailable",
    "autofocus": True,
    "auto_light_correction": True,
    "audio_in_observation": False,
    "processing": "decoded RGB, aspect-preserving area downsample",
    "simulation_mode": "deterministic_render",
}


def camera_model() -> dict[str, object]:
    """Return an owned copy of the public camera calibration."""

    return copy.deepcopy(CAMERA_MODEL)


def validated_rgb_frame(frame: np.ndarray) -> RgbFrame:
    """Validate the exact public camera byte contract without coercion."""

    data = np.asarray(frame)
    if data.shape != FRAME_SHAPE:
        raise ValueError(
            f"expected RGB camera shape {FRAME_SHAPE!r}, got {data.shape!r}"
        )
    if data.dtype != np.uint8:
        raise ValueError(f"expected RGB camera dtype uint8, got {data.dtype}")
    return data


def frame_record(frame: np.ndarray) -> tuple[str, str]:
    """Return the SHA-256 and base64 of exact row-major RGB camera bytes."""

    raw = validated_rgb_frame(frame).tobytes(order="C")
    return (
        hashlib.sha256(raw).hexdigest(),
        base64.b64encode(raw).decode("ascii"),
    )


def operational_telemetry(state: WorldState) -> dict[str, object]:
    """Project state into realistic, explicitly separated I/O products."""

    robot = state.robot
    arm_response_time = state.simulation_time_s + 0.008
    camera_sequence = max(0, int(arm_response_time / C920S_FRAME_PERIOD_S))
    camera_time = camera_sequence * C920S_FRAME_PERIOD_S
    feedback = roarm_feedback(
        robot.joints,
        robot.gripper_aperture,
        robot.joint_load_raw,
        supply_centivolts=robot.supply_centivolts,
        torque_enabled=robot.torque_enabled,
    )
    command_x = (
        BASE_AXIS_X_OFFSET_M
        + robot.command_reach * math.cos(robot.command_azimuth)
    ) * 1000.0
    command_y = robot.command_reach * math.sin(robot.command_azimuth) * 1000.0
    return {
        "schema_version": OBSERVATION_SCHEMA_VERSION,
        "sensor_contract_id": SENSOR_CONTRACT_ID,
        "mode": "operational",
        "sample": {
            "sequence": state.action_count,
            "host_time_s": round(arm_response_time, 9),
            "arm_request_time_s": round(state.simulation_time_s + 0.002, 9),
            "arm_response_time_s": round(arm_response_time, 9),
            "camera_capture_time_s": round(camera_time, 9),
            "sensor_skew_ms": round((arm_response_time - camera_time) * 1000.0, 6),
        },
        "controller": {
            "selected_coordinate": robot.selected_axis.name.lower(),
            "last_action": state.last_action,
            "last_step_duration_s": round(state.last_step_duration_s, 9),
            "command_json": {
                "T": 104,
                "x": command_x,
                "y": command_y,
                "z": robot.command_height * 1000.0,
                "t": gripper_angle_from_aperture(robot.gripper_aperture),
                "spd": 0.25,
            },
            "interlocked": robot.last_motion_rejected,
        },
        "arm": {
            "device": ROARM_MODEL,
            "transport": "USB serial JSONL",
            "baud": ROARM_SERIAL_BAUD,
            "request": dict(ROARM_FEEDBACK_COMMAND),
            "feedback": feedback,
        },
        "camera": {
            "device": C920S_MODEL,
            "transport": C920S_TRANSPORT,
            "source_format": C920S_SOURCE_FORMAT,
            "source_shape": list(C920S_SOURCE_SHAPE),
            "source_fps": C920S_SOURCE_FPS,
            "sequence": camera_sequence,
            "capture_time_s": round(camera_time, 9),
            "observation_encoding": FRAME_ENCODING,
            "observation_shape": list(FRAME_SHAPE),
            "autofocus": True,
            "auto_light_correction": True,
            "audio_in_observation": False,
        },
    }


def validated_operational_telemetry(
    value: object,
) -> dict[str, object]:
    """Fail closed if a public operational packet changes or leaks fields."""

    if not isinstance(value, Mapping):
        raise ValueError("operational telemetry must be an object")
    fields = {"schema_version", "sensor_contract_id", "mode", "sample", "controller", "arm", "camera"}
    if set(value) != fields:
        raise ValueError("operational telemetry fields changed")
    if (
        value["schema_version"] != OBSERVATION_SCHEMA_VERSION
        or value["sensor_contract_id"] != SENSOR_CONTRACT_ID
        or value["mode"] != "operational"
    ):
        raise ValueError("operational telemetry identity is invalid")
    sample = value["sample"]
    controller = value["controller"]
    arm = value["arm"]
    camera = value["camera"]
    if not all(isinstance(item, Mapping) for item in (sample, controller, arm, camera)):
        raise ValueError("operational I/O sections must be objects")
    if set(sample) != {"sequence", "host_time_s", "arm_request_time_s", "arm_response_time_s", "camera_capture_time_s", "sensor_skew_ms"}:
        raise ValueError("operational sample timing is invalid")
    if set(controller) != {"selected_coordinate", "last_action", "last_step_duration_s", "command_json", "interlocked"}:
        raise ValueError("operational controller state is invalid")
    if controller["selected_coordinate"] not in {"azimuth", "reach", "height"}:
        raise ValueError("selected coordinate is invalid")
    if not isinstance(controller["last_action"], int) or controller["last_action"] not in range(7):
        raise ValueError("last action is invalid")
    if not isinstance(controller["interlocked"], bool):
        raise ValueError("interlock state is invalid")
    command = controller["command_json"]
    if not isinstance(command, Mapping) or set(command) != {"T", "x", "y", "z", "t", "spd"} or command["T"] != 104:
        raise ValueError("RoArm command JSON is invalid")
    if set(arm) != {"device", "transport", "baud", "request", "feedback"}:
        raise ValueError("RoArm section is invalid")
    feedback = arm["feedback"]
    expected_feedback = {"T", "x", "y", "z", "b", "s", "e", "t", "torB", "torS", "torE", "torH", "torswitchB", "torswitchS", "torswitchE", "torswitchH", "v"}
    if not isinstance(feedback, Mapping) or set(feedback) != expected_feedback or feedback["T"] != 1051:
        raise ValueError("RoArm T=1051 feedback is invalid")
    if set(camera) != {"device", "transport", "source_format", "source_shape", "source_fps", "sequence", "capture_time_s", "observation_encoding", "observation_shape", "autofocus", "auto_light_correction", "audio_in_observation"}:
        raise ValueError("C920s section is invalid")
    numeric_values = [*sample.values(), controller["last_step_duration_s"], *command.values(), *feedback.values(), camera["source_fps"], camera["sequence"], camera["capture_time_s"]]
    if any(isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(float(number)) for number in numeric_values):
        raise ValueError("operational telemetry contains a non-finite measure")
    if camera["source_shape"] != list(C920S_SOURCE_SHAPE) or camera["observation_shape"] != list(FRAME_SHAPE):
        raise ValueError("C920s frame geometry is invalid")
    if sample["sequence"] < 0 or camera["sequence"] < 0:
        raise ValueError("sensor sequence is invalid")
    return copy.deepcopy(dict(value))


def calibration_telemetry(state: CalibrationState) -> dict[str, object]:
    """Return the Phase-0 command shell's separate public telemetry packet."""

    azimuth, reach, height = state.command_ticks()
    return {
        "schema_version": OBSERVATION_SCHEMA_VERSION,
        "sensor_contract_id": "rb01-calibration-telemetry-v2",
        "mode": "calibration",
        "turn": state.turns,
        "simulation_time_s": round(state.turns * 0.25, 9),
        "selected_coordinate": state.selected.name.lower(),
        "command": {
            "azimuth_rad": azimuth * AZIMUTH_STEP_DEG * pi / 180.0,
            "reach_delta_m": reach * REACH_STEP_M,
            "height_delta_m": height * HEIGHT_STEP_M,
        },
        "measured": {
            "command_ticks": [azimuth, reach, height],
        },
        "gripper": {
            "open": state.gripper_open,
            "aperture_m": 0.080 if state.gripper_open else 0.008,
        },
        "contact_load_normalized": 0.0,
        "motion": {
            "rejected": state.rejected,
            "reason": "command_bounds" if state.rejected else "",
        },
        "last_action": state.last_action,
    }


__all__ = [
    "CAMERA_MODEL",
    "FRAME_ENCODING",
    "FRAME_SHAPE",
    "OBSERVATION_SCHEMA_VERSION",
    "RgbFrame",
    "SENSOR_CONTRACT_ID",
    "calibration_telemetry",
    "camera_model",
    "frame_record",
    "operational_telemetry",
    "validated_operational_telemetry",
    "validated_rgb_frame",
]

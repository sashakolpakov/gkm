"""Transport boundary for a physical RoArm-M2-S plus Logitech C920s.

No serial or camera library is imposed on the simulator.  Production adapters
only need to implement the two small protocols below (for example pyserial and
OpenCV/V4L2).  This layer performs the device-independent validation and pairs
the most recent UVC frame with one host-timestamped T=1051 arm response.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Callable, Mapping, Protocol, runtime_checkable

import numpy as np

from .hardware import (
    C920S_FRAME_PERIOD_S,
    ROARM_FEEDBACK_COMMAND,
    ROARM_FEEDBACK_TYPE,
)


class DeviceIOError(RuntimeError):
    """A physical transport returned malformed, stale, or mismatched data."""


@runtime_checkable
class RoArmJsonTransport(Protocol):
    def exchange(self, command: Mapping[str, int | float]) -> Mapping[str, object]:
        """Write one newline-delimited JSON command and return one response."""


@dataclass(frozen=True, slots=True)
class UvcFrame:
    sequence: int
    capture_time_s: float
    image: np.ndarray


@runtime_checkable
class C920sFrameSource(Protocol):
    def latest(self) -> UvcFrame:
        """Return the newest decoded BGR or RGB frame and its capture stamp."""


@dataclass(frozen=True, slots=True)
class DeviceSample:
    sequence: int
    arm_request_time_s: float
    arm_response_time_s: float
    camera_sequence: int
    camera_capture_time_s: float
    sensor_skew_ms: float
    arm_feedback: dict[str, object]
    frame: np.ndarray


_FEEDBACK_FIELDS = {
    "T", "x", "y", "z", "b", "s", "e", "t",
    "torB", "torS", "torE", "torH",
    "torswitchB", "torswitchS", "torswitchE", "torswitchH", "v",
}


def validated_roarm_feedback(value: Mapping[str, object]) -> dict[str, object]:
    if set(value) != _FEEDBACK_FIELDS or value.get("T") != ROARM_FEEDBACK_TYPE:
        raise DeviceIOError("expected a complete RoArm T=1051 feedback object")
    if any(
        isinstance(item, bool)
        or not isinstance(item, (int, float))
        or not math.isfinite(float(item))
        for item in value.values()
    ):
        raise DeviceIOError("RoArm feedback contains a non-finite numeric field")
    for field in ("torswitchB", "torswitchS", "torswitchE", "torswitchH"):
        if value[field] not in (0, 1):
            raise DeviceIOError(f"RoArm {field} must be 0 or 1")
    if not 700 <= float(value["v"]) <= 1300:
        raise DeviceIOError("RoArm supply voltage is outside its 7-13 V input range")
    return copy.deepcopy(dict(value))


class ReferenceDeviceIO:
    """Acquire correlated samples without claiming hardware synchronization."""

    def __init__(
        self,
        arm: RoArmJsonTransport,
        camera: C920sFrameSource,
        *,
        monotonic: Callable[[], float],
        maximum_camera_age_s: float = 2.0 * C920S_FRAME_PERIOD_S,
    ) -> None:
        if maximum_camera_age_s <= 0.0 or not math.isfinite(maximum_camera_age_s):
            raise ValueError("maximum_camera_age_s must be finite and positive")
        self._arm = arm
        self._camera = camera
        self._clock = monotonic
        self._maximum_camera_age_s = maximum_camera_age_s
        self._sequence = 0
        self._last_camera_sequence = -1

    def sample(self) -> DeviceSample:
        request_time = float(self._clock())
        response = validated_roarm_feedback(
            self._arm.exchange(ROARM_FEEDBACK_COMMAND)
        )
        response_time = float(self._clock())
        capture = self._camera.latest()
        if capture.sequence < self._last_camera_sequence:
            raise DeviceIOError("C920s frame sequence moved backwards")
        if capture.image.ndim != 3 or capture.image.shape[2] != 3 or capture.image.dtype != np.uint8:
            raise DeviceIOError("C920s frame must be HxWx3 uint8")
        age = response_time - float(capture.capture_time_s)
        if age < -C920S_FRAME_PERIOD_S:
            raise DeviceIOError("C920s capture timestamp is in the future")
        if age > self._maximum_camera_age_s:
            raise DeviceIOError("C920s frame is stale relative to arm feedback")
        self._last_camera_sequence = capture.sequence
        sample = DeviceSample(
            sequence=self._sequence,
            arm_request_time_s=request_time,
            arm_response_time_s=response_time,
            camera_sequence=int(capture.sequence),
            camera_capture_time_s=float(capture.capture_time_s),
            sensor_skew_ms=age * 1000.0,
            arm_feedback=response,
            frame=capture.image.copy(),
        )
        self._sequence += 1
        return sample

    def send(self, command: Mapping[str, int | float]) -> dict[str, object]:
        """Send a documented RoArm command; transport-specific acks pass through."""

        command_type = command.get("T")
        if isinstance(command_type, bool) or not isinstance(command_type, int):
            raise DeviceIOError("RoArm command requires an integer T field")
        return copy.deepcopy(dict(self._arm.exchange(command)))


__all__ = [
    "C920sFrameSource",
    "DeviceIOError",
    "DeviceSample",
    "ReferenceDeviceIO",
    "RoArmJsonTransport",
    "UvcFrame",
    "validated_roarm_feedback",
]

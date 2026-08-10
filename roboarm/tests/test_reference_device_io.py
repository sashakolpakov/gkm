from __future__ import annotations

import numpy as np
import pytest

from roboarm_game.device_io import (
    DeviceIOError,
    ReferenceDeviceIO,
    UvcFrame,
    validated_roarm_feedback,
)
from roboarm_game.hardware import (
    ROARM_ENCODER_STEP_RAD,
    quantize_encoder,
    roarm_feedback,
)


class FakeArm:
    def __init__(self, response: dict[str, object]) -> None:
        self.response = response
        self.commands: list[dict[str, int | float]] = []

    def exchange(self, command):
        self.commands.append(dict(command))
        return dict(self.response)


class FakeCamera:
    def __init__(self, capture_time_s: float = 1.0) -> None:
        self.capture_time_s = capture_time_s
        self.sequence = 30

    def latest(self) -> UvcFrame:
        return UvcFrame(
            sequence=self.sequence,
            capture_time_s=self.capture_time_s,
            image=np.zeros((720, 1280, 3), dtype=np.uint8),
        )


def feedback() -> dict[str, object]:
    return roarm_feedback(
        (0.0, 0.1, 1.5),
        0.04,
        (8, 120, 70, 180),
    )


def test_encoder_and_feedback_match_reference_resolution_and_field_family() -> None:
    angle = 0.12345
    quantized = quantize_encoder(angle)
    assert abs(angle - quantized) <= ROARM_ENCODER_STEP_RAD / 2.0
    packet = validated_roarm_feedback(feedback())
    assert packet["T"] == 1051
    assert packet["v"] == 1200
    assert {"b", "s", "e", "t", "torB", "torS", "torE", "torH"} <= packet.keys()
    assert not {"object_attached", "contact_force", "depth", "collision_reason"} & packet.keys()


def test_reference_io_pairs_independent_arm_and_webcam_timestamps() -> None:
    times = iter((1.020, 1.028))
    arm = FakeArm(feedback())
    camera = FakeCamera(capture_time_s=1.000)
    connector = ReferenceDeviceIO(arm, camera, monotonic=lambda: next(times))

    sample = connector.sample()

    assert arm.commands == [{"T": 105}]
    assert sample.arm_response_time_s == 1.028
    assert sample.camera_capture_time_s == 1.0
    assert sample.sensor_skew_ms == pytest.approx(28.0)
    assert sample.frame.shape == (720, 1280, 3)


def test_reference_io_rejects_a_stale_webcam_frame() -> None:
    times = iter((2.0, 2.01))
    connector = ReferenceDeviceIO(
        FakeArm(feedback()),
        FakeCamera(capture_time_s=1.0),
        monotonic=lambda: next(times),
    )
    with pytest.raises(DeviceIOError, match="stale"):
        connector.sample()

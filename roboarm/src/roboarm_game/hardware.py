"""Reference-device I/O constants and deterministic sensor projections.

The public RoArm packet mirrors Waveshare ``CMD_SERVO_RAD_FEEDBACK`` (T=1051).
Cartesian coordinates are firmware-computed from encoder angles; they are not
independent position measurements.  The C920s is a separate USB/UVC clock
domain and therefore carries its own capture sequence and host timestamp.
"""

from __future__ import annotations

from math import pi
from typing import Final

from .kinematics import JointVector, exact_anchors


ROARM_MODEL: Final[str] = "Waveshare RoArm-M2-S"
ROARM_FEEDBACK_COMMAND: Final[dict[str, int]] = {"T": 105}
ROARM_FEEDBACK_TYPE: Final[int] = 1051
ROARM_SERIAL_BAUD: Final[int] = 115_200
ROARM_ENCODER_COUNTS: Final[int] = 4096
ROARM_ENCODER_STEP_RAD: Final[float] = 2.0 * pi / ROARM_ENCODER_COUNTS
ROARM_SUPPLY_CENTIVOLTS: Final[int] = 1200

C920S_MODEL: Final[str] = "Logitech C920s Pro HD"
C920S_TRANSPORT: Final[str] = "USB UVC"
C920S_SOURCE_FORMAT: Final[str] = "MJPG"
C920S_SOURCE_SHAPE: Final[tuple[int, int, int]] = (1080, 1920, 3)
C920S_SOURCE_FPS: Final[int] = 30
C920S_DIAGONAL_FOV_DEG: Final[float] = 78.0
C920S_FRAME_PERIOD_S: Final[float] = 1.0 / C920S_SOURCE_FPS


def quantize_encoder(angle: float) -> float:
    """Return the angle visible through a 12-bit absolute encoder."""

    return round(angle / ROARM_ENCODER_STEP_RAD) * ROARM_ENCODER_STEP_RAD


def quantize_joints(joints: JointVector) -> JointVector:
    return tuple(quantize_encoder(value) for value in joints)  # type: ignore[return-value]


def gripper_angle_from_aperture(aperture_m: float) -> float:
    """Map the simulated parallel-jaw aperture to the clamp servo angle.

    The stock clamp uses approximately 1.08 rad when open and 3.14 rad when
    closed.  The aperture is simulator geometry, while the returned angle is
    the quantity the physical bus-servo encoder would report.
    """

    from .config import GRIPPER_CLOSED_APERTURE_M, GRIPPER_OPEN_APERTURE_M

    fraction = (
        (aperture_m - GRIPPER_CLOSED_APERTURE_M)
        / (GRIPPER_OPEN_APERTURE_M - GRIPPER_CLOSED_APERTURE_M)
    )
    fraction = min(1.0, max(0.0, fraction))
    return quantize_encoder(3.14 - fraction * (3.14 - 1.08))


def roarm_feedback(
    joints: JointVector,
    aperture_m: float,
    joint_load_raw: tuple[int, int, int, int],
    *,
    supply_centivolts: int = ROARM_SUPPLY_CENTIVOLTS,
    torque_enabled: tuple[bool, bool, bool, bool] = (True, True, True, True),
) -> dict[str, int | float]:
    """Build the exact field family returned by stock feedback command 105."""

    measured = quantize_joints(joints)
    tcp = exact_anchors(measured).tcp
    hand = gripper_angle_from_aperture(aperture_m)
    base, shoulder, elbow = measured
    load_b, load_s, load_e, load_h = joint_load_raw
    torque_b, torque_s, torque_e, torque_h = torque_enabled
    return {
        "T": ROARM_FEEDBACK_TYPE,
        "x": tcp[0] * 1000.0,
        "y": tcp[1] * 1000.0,
        "z": tcp[2] * 1000.0,
        "b": base,
        "s": shoulder,
        "e": elbow,
        "t": hand,
        "torB": int(load_b),
        "torS": int(load_s),
        "torE": int(load_e),
        "torH": int(load_h),
        "torswitchB": int(torque_b),
        "torswitchS": int(torque_s),
        "torswitchE": int(torque_e),
        "torswitchH": int(torque_h),
        "v": int(supply_centivolts),
    }


__all__ = [
    "C920S_DIAGONAL_FOV_DEG",
    "C920S_FRAME_PERIOD_S",
    "C920S_MODEL",
    "C920S_SOURCE_FORMAT",
    "C920S_SOURCE_FPS",
    "C920S_SOURCE_SHAPE",
    "C920S_TRANSPORT",
    "ROARM_ENCODER_COUNTS",
    "ROARM_ENCODER_STEP_RAD",
    "ROARM_FEEDBACK_COMMAND",
    "ROARM_FEEDBACK_TYPE",
    "ROARM_MODEL",
    "ROARM_SERIAL_BAUD",
    "gripper_angle_from_aperture",
    "quantize_encoder",
    "quantize_joints",
    "roarm_feedback",
]

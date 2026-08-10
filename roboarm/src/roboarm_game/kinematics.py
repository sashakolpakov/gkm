"""Exact Xacro transforms and deterministic cylindrical inverse kinematics."""

from __future__ import annotations

from dataclasses import dataclass
from math import acos, atan2, ceil, cos, hypot, pi, sin

import numpy as np
from numpy.typing import NDArray

from .config import (
    BASE_AXIS_X_OFFSET_M,
    FOREARM_TCP_X_M,
    FOREARM_TCP_Z_M,
    FOREARM_TO_TCP_EFFECTIVE_M,
    KINEMATIC_TOLERANCE_M,
    Q0_LIMITS,
    Q1_LIMITS,
    Q2_LIMITS,
    SHOULDER_HEIGHT_M,
    UPPER_ARM_EFFECTIVE_M,
    UPPER_ARM_RADIAL_OFFSET_M,
    UPPER_ARM_X_M,
)

Matrix4 = NDArray[np.float64]
Vector3 = tuple[float, float, float]
JointVector = tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class ArmAnchors:
    base: Vector3
    shoulder: Vector3
    elbow: Vector3
    wrist: Vector3
    tcp: Vector3


@dataclass(frozen=True, slots=True)
class IKResult:
    joints: JointVector
    tcp: Vector3
    error_m: float


def rotation_x(angle: float) -> Matrix4:
    result = np.eye(4, dtype=np.float64)
    cosine, sine = cos(angle), sin(angle)
    result[1:3, 1:3] = ((cosine, -sine), (sine, cosine))
    return result


def rotation_y(angle: float) -> Matrix4:
    result = np.eye(4, dtype=np.float64)
    cosine, sine = cos(angle), sin(angle)
    result[(0, 0, 2, 2), (0, 2, 0, 2)] = (cosine, sine, -sine, cosine)
    return result


def rotation_z(angle: float) -> Matrix4:
    result = np.eye(4, dtype=np.float64)
    cosine, sine = cos(angle), sin(angle)
    result[0:2, 0:2] = ((cosine, -sine), (sine, cosine))
    return result


def translation(x: float, y: float, z: float) -> Matrix4:
    result = np.eye(4, dtype=np.float64)
    result[:3, 3] = (x, y, z)
    return result


def rpy(roll: float, pitch: float, yaw: float) -> Matrix4:
    return rotation_z(yaw) @ rotation_y(pitch) @ rotation_x(roll)


def _point(transform: Matrix4) -> Vector3:
    vector = transform[:3, 3]
    return float(vector[0]), float(vector[1]), float(vector[2])


def exact_transforms(joints: JointVector) -> dict[str, Matrix4]:
    """Return the authoritative parent-to-world transforms from the Xacro."""

    q0, q1, q2 = joints
    world = np.eye(4, dtype=np.float64)
    base = world
    link1 = (
        base
        @ translation(BASE_AXIS_X_OFFSET_M, 0.0, SHOULDER_HEIGHT_M)
        @ rotation_z(q0)
    )
    link2 = link1 @ rpy(-1.5708, -1.5708, 0.0) @ rotation_z(q1)
    link3 = (
        link2
        @ translation(UPPER_ARM_X_M, UPPER_ARM_RADIAL_OFFSET_M, 0.0)
        @ rpy(0.0, 0.0, 1.5708)
        @ rotation_z(q2)
    )
    gripper = (
        link3
        @ translation(0.002906, -0.21599, -0.00066683)
        @ rpy(-1.5708, 0.0, -1.5708)
    )
    tcp = (
        link3
        @ translation(FOREARM_TCP_X_M, -FOREARM_TCP_Z_M, 0.0)
        @ rpy(1.5708, 0.0, -1.5708)
    )
    return {
        "world": world,
        "base_link": base,
        "link1": link1,
        "link2": link2,
        "link3": link3,
        "gripper_link": gripper,
        "hand_tcp": tcp,
    }


def exact_anchors(joints: JointVector) -> ArmAnchors:
    transforms = exact_transforms(joints)
    elbow = _point(transforms["link3"])
    wrist = _point(transforms["gripper_link"])
    return ArmAnchors(
        base=(0.0, 0.0, 0.0),
        shoulder=_point(transforms["link1"]),
        elbow=elbow,
        wrist=wrist,
        tcp=_point(transforms["hand_tcp"]),
    )


def cylindrical_from_tcp(tcp: Vector3) -> tuple[float, float, float]:
    relative_x = tcp[0] - BASE_AXIS_X_OFFSET_M
    azimuth = atan2(tcp[1], relative_x)
    reach = hypot(relative_x, tcp[1])
    return azimuth, reach, tcp[2]


_UPPER_OFFSET = atan2(UPPER_ARM_RADIAL_OFFSET_M, UPPER_ARM_X_M)
_FOREARM_OFFSET = atan2(FOREARM_TCP_X_M, FOREARM_TCP_Z_M)
_EXACT_AZIMUTH_BIAS = cylindrical_from_tcp(exact_anchors((0.0, 0.0, 0.0)).tcp)[0]


def _within(value: float, limits: tuple[float, float], tolerance: float = 1e-9) -> bool:
    return limits[0] - tolerance <= value <= limits[1] + tolerance


def _analytic_seed(reach: float, height: float) -> tuple[float, float] | None:
    vertical = height - SHOULDER_HEIGHT_M
    numerator = (
        reach * reach
        + vertical * vertical
        - UPPER_ARM_EFFECTIVE_M * UPPER_ARM_EFFECTIVE_M
        - FOREARM_TO_TCP_EFFECTIVE_M * FOREARM_TO_TCP_EFFECTIVE_M
    )
    denominator = 2.0 * UPPER_ARM_EFFECTIVE_M * FOREARM_TO_TCP_EFFECTIVE_M
    cosine_delta = numerator / denominator
    if cosine_delta < -1.0 - 1e-12 or cosine_delta > 1.0 + 1e-12:
        return None
    cosine_delta = min(1.0, max(-1.0, cosine_delta))
    delta = acos(cosine_delta)
    theta1 = atan2(reach, vertical) - atan2(
        FOREARM_TO_TCP_EFFECTIVE_M * sin(delta),
        UPPER_ARM_EFFECTIVE_M
        + FOREARM_TO_TCP_EFFECTIVE_M * cos(delta),
    )
    q1 = theta1 - _UPPER_OFFSET
    q2 = delta + _UPPER_OFFSET - _FOREARM_OFFSET
    return q1, q2


def solve_cylindrical(
    azimuth: float,
    reach: float,
    height: float,
) -> IKResult | None:
    """Solve the fixed positive-elbow branch and refine against exact Xacro FK."""

    seed = _analytic_seed(reach, height)
    if seed is None:
        return None
    q1, q2 = seed
    q0 = azimuth - _EXACT_AZIMUTH_BIAS

    for _ in range(8):
        tcp = exact_anchors((q0, q1, q2)).tcp
        observed_azimuth, observed_reach, observed_height = cylindrical_from_tcp(tcp)
        residual = np.array(
            (reach - observed_reach, height - observed_height),
            dtype=np.float64,
        )
        if float(np.linalg.norm(residual)) <= KINEMATIC_TOLERANCE_M * 0.1:
            q0 += azimuth - observed_azimuth
            break
        epsilon = 1.0e-6
        jacobian = np.empty((2, 2), dtype=np.float64)
        for column, candidate in enumerate(((q1 + epsilon, q2), (q1, q2 + epsilon))):
            shifted = exact_anchors((q0, candidate[0], candidate[1])).tcp
            _, shifted_reach, shifted_height = cylindrical_from_tcp(shifted)
            jacobian[:, column] = (
                (shifted_reach - observed_reach) / epsilon,
                (shifted_height - observed_height) / epsilon,
            )
        try:
            correction = np.linalg.solve(jacobian, residual)
        except np.linalg.LinAlgError:
            return None
        q1 += float(correction[0])
        q2 += float(correction[1])
        q0 += azimuth - observed_azimuth

    joints = (q0, q1, q2)
    if not (
        _within(q0, Q0_LIMITS)
        and _within(q1, Q1_LIMITS)
        and _within(q2, Q2_LIMITS)
    ):
        return None
    tcp = exact_anchors(joints).tcp
    observed_azimuth, observed_reach, observed_height = cylindrical_from_tcp(tcp)
    angular_error = abs(
        atan2(
            sin(observed_azimuth - azimuth),
            cos(observed_azimuth - azimuth),
        )
    )
    error = hypot(observed_reach - reach, observed_height - height)
    error = hypot(error, angular_error * max(reach, 1.0e-6))
    if error > KINEMATIC_TOLERANCE_M:
        return None
    return IKResult(joints=joints, tcp=tcp, error_m=error)


def interpolate_joints(
    start: JointVector,
    end: JointVector,
    max_delta: float,
) -> tuple[JointVector, ...]:
    largest = max(abs(end_value - start_value) for start_value, end_value in zip(start, end))
    steps = max(1, int(ceil(largest / max_delta)))
    return tuple(
        tuple(
            start_value + (end_value - start_value) * index / steps
            for start_value, end_value in zip(start, end)
        )
        for index in range(1, steps + 1)
    )


__all__ = [
    "ArmAnchors",
    "IKResult",
    "JointVector",
    "cylindrical_from_tcp",
    "exact_anchors",
    "exact_transforms",
    "interpolate_joints",
    "solve_cylindrical",
]

"""Versioned physical and canonical-scene configuration."""

from __future__ import annotations

from math import hypot, pi
from typing import Final


SCENE_SCHEMA_VERSION: Final[int] = 2
CANONICAL_SCENE_ID: Final[str] = "pick-place-v2"
ROUND1_ACTION_BUDGET: Final[int] = 160

BASE_AXIS_X_OFFSET_M: Final[float] = 0.0100000008759151
SHOULDER_HEIGHT_M: Final[float] = 0.123059270461044
UPPER_ARM_X_M: Final[float] = 0.236815132922094
UPPER_ARM_RADIAL_OFFSET_M: Final[float] = 0.0300023995170449
UPPER_ARM_EFFECTIVE_M: Final[float] = hypot(
    UPPER_ARM_X_M,
    UPPER_ARM_RADIAL_OFFSET_M,
)
FOREARM_TCP_X_M: Final[float] = 0.002
FOREARM_TCP_Z_M: Final[float] = 0.2802
FOREARM_TO_TCP_EFFECTIVE_M: Final[float] = hypot(
    FOREARM_TCP_Z_M,
    FOREARM_TCP_X_M,
)

Q0_LIMITS: Final[tuple[float, float]] = (-3.1416, 3.1416)
Q1_LIMITS: Final[tuple[float, float]] = (-1.5708, 1.5708)
Q2_LIMITS: Final[tuple[float, float]] = (-1.0, 2.95)
QG_LIMITS: Final[tuple[float, float]] = (0.0, 1.5)

SERVO_NO_LOAD_MAX_RAD_S: Final[float] = 40.0 * 2.0 * pi / 60.0
SERVO_ACCELERATION_RAD_S2: Final[float] = 1000.0 * 2.0 * pi / 4096.0
SERVO_SETTLING_TIME_S: Final[float] = 0.040
RATED_PAYLOAD_KG: Final[float] = 0.5
RATED_PAYLOAD_REACH_M: Final[float] = 0.5
RATED_PAYLOAD_MOMENT_NM: Final[float] = (
    RATED_PAYLOAD_KG * 9.80665 * RATED_PAYLOAD_REACH_M
)

COMMAND_AZIMUTH_LIMITS_RAD: Final[tuple[float, float]] = (
    -80.0 * pi / 180.0,
    80.0 * pi / 180.0,
)
COMMAND_REACH_LIMITS_M: Final[tuple[float, float]] = (0.16, 0.48)
COMMAND_HEIGHT_LIMITS_M: Final[tuple[float, float]] = (0.045, 0.52)

INITIAL_AZIMUTH_RAD: Final[float] = 0.0
INITIAL_REACH_M: Final[float] = 0.30
INITIAL_HEIGHT_M: Final[float] = 0.27

TABLE_Z_M: Final[float] = 0.0
TABLE_SIZE_M: Final[tuple[float, float]] = (0.90, 0.90)
BASE_RADIUS_M: Final[float] = 0.065
BASE_HEIGHT_M: Final[float] = 0.035
BASE_COLUMN_RADIUS_M: Final[float] = 0.026
UPPER_ARM_RADIUS_M: Final[float] = 0.026
FOREARM_RADIUS_M: Final[float] = 0.021
WRIST_LINK_RADIUS_M: Final[float] = 0.017
SHOULDER_JOINT_RADIUS_M: Final[float] = 0.035
ELBOW_JOINT_RADIUS_M: Final[float] = 0.031
WRIST_JOINT_RADIUS_M: Final[float] = 0.024
ROBOT_LINK_RADIUS_M: Final[float] = UPPER_ARM_RADIUS_M
GRIPPER_RADIUS_M: Final[float] = 0.018
GRIPPER_OPEN_APERTURE_M: Final[float] = 0.080
GRIPPER_CLOSED_APERTURE_M: Final[float] = 0.008
GRIPPER_JAW_DEPTH_M: Final[float] = 0.065
GRIPPER_VERTICAL_TOLERANCE_M: Final[float] = 0.030
GRIPPER_PALM_SIZE_M: Final[tuple[float, float, float]] = (
    0.060,
    0.085,
    0.024,
)
GRIPPER_PALM_RADIAL_OFFSET_M: Final[float] = -0.026
GRIPPER_JAW_SIZE_M: Final[tuple[float, float, float]] = (
    0.060,
    0.009,
    0.040,
)
GRIPPER_JAW_RADIAL_OFFSET_M: Final[float] = 0.004
GRIPPER_JAW_VERTICAL_OFFSET_M: Final[float] = -0.020
SWEEP_MAX_JOINT_DELTA_RAD: Final[float] = 0.025

OBJECT_SIZE_M: Final[tuple[float, float, float]] = (0.040, 0.040, 0.050)
OBJECT_MASS_KG: Final[float] = 0.080
OBJECT_START_AZIMUTH_RAD: Final[float] = 0.0
OBJECT_START_REACH_M: Final[float] = 0.30

TARGET_AZIMUTH_RAD: Final[float] = 60.0 * pi / 180.0
TARGET_REACH_M: Final[float] = 0.30
TARGET_SIZE_M: Final[tuple[float, float, float]] = (0.14, 0.14, 0.018)
TARGET_WALL_HEIGHT_M: Final[float] = 0.048
TARGET_WALL_THICKNESS_M: Final[float] = 0.006

BARRIER_AZIMUTH_RAD: Final[float] = 20.0 * pi / 180.0
BARRIER_REACH_M: Final[float] = 0.30
BARRIER_SIZE_M: Final[tuple[float, float, float]] = (0.075, 0.090, 0.125)
BARRIER_CAP_OVERHANG_M: Final[float] = 0.003
BARRIER_CAP_THICKNESS_M: Final[float] = 0.009

WORKCELL_REAR_WALL_CENTER_M: Final[tuple[float, float, float]] = (
    0.0,
    0.63,
    0.315,
)
WORKCELL_REAR_WALL_SIZE_M: Final[tuple[float, float, float]] = (
    1.55,
    0.035,
    0.85,
)
WORKCELL_POST_X_M: Final[tuple[float, float]] = (-0.48, 0.48)
WORKCELL_POST_Y_M: Final[float] = 0.48
WORKCELL_POST_SIZE_M: Final[tuple[float, float, float]] = (
    0.018,
    0.018,
    0.60,
)

KINEMATIC_TOLERANCE_M: Final[float] = 2.0e-5

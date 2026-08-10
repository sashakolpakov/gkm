"""Physical solids shared by mechanics and the authoritative RGB renderer."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, sin

from .config import (
    BARRIER_CAP_OVERHANG_M,
    BARRIER_CAP_THICKNESS_M,
    BASE_COLUMN_RADIUS_M,
    BASE_HEIGHT_M,
    FOREARM_RADIUS_M,
    GRIPPER_JAW_RADIAL_OFFSET_M,
    GRIPPER_JAW_SIZE_M,
    GRIPPER_JAW_VERTICAL_OFFSET_M,
    GRIPPER_PALM_RADIAL_OFFSET_M,
    GRIPPER_PALM_SIZE_M,
    TARGET_WALL_HEIGHT_M,
    TARGET_WALL_THICKNESS_M,
    UPPER_ARM_RADIUS_M,
    WORKCELL_POST_SIZE_M,
    WORKCELL_POST_X_M,
    WORKCELL_POST_Y_M,
    WORKCELL_REAR_WALL_CENTER_M,
    WORKCELL_REAR_WALL_SIZE_M,
    WRIST_LINK_RADIUS_M,
)
from .kinematics import ArmAnchors, Vector3
from .world_state import SceneBox


@dataclass(frozen=True, slots=True)
class Capsule:
    body_id: str
    start: Vector3
    end: Vector3
    radius: float


@dataclass(frozen=True, slots=True)
class YawBox:
    body_id: str
    center: Vector3
    size: Vector3
    yaw: float


def barrier_cap(barrier: SceneBox) -> SceneBox:
    return SceneBox(
        box_id="barrier-cap",
        center=(
            barrier.center[0],
            barrier.center[1],
            barrier.center[2]
            + barrier.size[2] * 0.5
            + BARRIER_CAP_THICKNESS_M * 0.5,
        ),
        size=(
            barrier.size[0] + BARRIER_CAP_OVERHANG_M * 2.0,
            barrier.size[1] + BARRIER_CAP_OVERHANG_M * 2.0,
            BARRIER_CAP_THICKNESS_M,
        ),
        color_role=barrier.color_role,
    )


def target_walls(target: SceneBox) -> tuple[SceneBox, ...]:
    half_x = target.size[0] * 0.5
    half_y = target.size[1] * 0.5
    center_z = TARGET_WALL_HEIGHT_M * 0.5
    return (
        SceneBox(
            box_id="target-wall-y-minus",
            center=(target.center[0], target.center[1] - half_y, center_z),
            size=(
                target.size[0],
                TARGET_WALL_THICKNESS_M,
                TARGET_WALL_HEIGHT_M,
            ),
            color_role=target.color_role,
        ),
        SceneBox(
            box_id="target-wall-y-plus",
            center=(target.center[0], target.center[1] + half_y, center_z),
            size=(
                target.size[0],
                TARGET_WALL_THICKNESS_M,
                TARGET_WALL_HEIGHT_M,
            ),
            color_role=target.color_role,
        ),
        SceneBox(
            box_id="target-wall-x-minus",
            center=(target.center[0] - half_x, target.center[1], center_z),
            size=(
                TARGET_WALL_THICKNESS_M,
                target.size[1],
                TARGET_WALL_HEIGHT_M,
            ),
            color_role=target.color_role,
        ),
        SceneBox(
            box_id="target-wall-x-plus",
            center=(target.center[0] + half_x, target.center[1], center_z),
            size=(
                TARGET_WALL_THICKNESS_M,
                target.size[1],
                TARGET_WALL_HEIGHT_M,
            ),
            color_role=target.color_role,
        ),
    )


def workcell_solids() -> tuple[SceneBox, ...]:
    solids = [
        SceneBox(
            box_id="workcell-rear-wall",
            center=WORKCELL_REAR_WALL_CENTER_M,
            size=WORKCELL_REAR_WALL_SIZE_M,
            color_role=0,
        )
    ]
    for index, x in enumerate(WORKCELL_POST_X_M):
        solids.append(
            SceneBox(
                box_id=f"workcell-safety-post-{index + 1}",
                center=(x, WORKCELL_POST_Y_M, WORKCELL_POST_SIZE_M[2] * 0.5),
                size=WORKCELL_POST_SIZE_M,
                color_role=0,
            )
        )
    return tuple(solids)


def robot_capsules(anchors: ArmAnchors) -> tuple[Capsule, ...]:
    return (
        Capsule(
            "base-column",
            (0.0, 0.0, BASE_HEIGHT_M),
            anchors.shoulder,
            BASE_COLUMN_RADIUS_M,
        ),
        Capsule(
            "upper-arm",
            anchors.shoulder,
            anchors.elbow,
            UPPER_ARM_RADIUS_M,
        ),
        Capsule(
            "forearm",
            anchors.elbow,
            anchors.wrist,
            FOREARM_RADIUS_M,
        ),
        Capsule(
            "wrist-link",
            anchors.wrist,
            anchors.tcp,
            WRIST_LINK_RADIUS_M,
        ),
    )


def _offset_point(
    origin: Vector3,
    yaw: float,
    radial: float,
    tangential: float,
    vertical: float,
) -> Vector3:
    cosine = cos(yaw)
    sine = sin(yaw)
    return (
        origin[0] + radial * cosine - tangential * sine,
        origin[1] + radial * sine + tangential * cosine,
        origin[2] + vertical,
    )


def gripper_boxes(
    tcp: Vector3,
    yaw: float,
    aperture: float,
) -> tuple[YawBox, ...]:
    palm = YawBox(
        "gripper-palm",
        _offset_point(
            tcp,
            yaw,
            GRIPPER_PALM_RADIAL_OFFSET_M,
            0.0,
            0.0,
        ),
        GRIPPER_PALM_SIZE_M,
        yaw,
    )
    jaws = tuple(
        YawBox(
            f"gripper-jaw-{'minus' if direction < 0 else 'plus'}",
            _offset_point(
                tcp,
                yaw,
                GRIPPER_JAW_RADIAL_OFFSET_M,
                aperture * 0.5 * direction,
                GRIPPER_JAW_VERTICAL_OFFSET_M,
            ),
            GRIPPER_JAW_SIZE_M,
            yaw,
        )
        for direction in (-1.0, 1.0)
    )
    return (palm, *jaws)


def attachment_local_offset(
    tcp: Vector3,
    yaw: float,
    object_center: Vector3,
) -> Vector3:
    """Return the object center in the rotating gripper yaw frame."""

    delta_x = object_center[0] - tcp[0]
    delta_y = object_center[1] - tcp[1]
    cosine = cos(yaw)
    sine = sin(yaw)
    return (
        delta_x * cosine + delta_y * sine,
        -delta_x * sine + delta_y * cosine,
        object_center[2] - tcp[2],
    )


def attached_world_position(
    tcp: Vector3,
    yaw: float,
    local_offset: Vector3,
) -> Vector3:
    return _offset_point(
        tcp,
        yaw,
        local_offset[0],
        local_offset[1],
        local_offset[2],
    )


__all__ = [
    "Capsule",
    "YawBox",
    "attached_world_position",
    "attachment_local_offset",
    "barrier_cap",
    "gripper_boxes",
    "robot_capsules",
    "target_walls",
    "workcell_solids",
]

"""Deepcopy-safe state for the operational pick-and-place vertical slice."""

from __future__ import annotations

from dataclasses import dataclass, field

from .interface import Coordinate
from .kinematics import JointVector, Vector3


@dataclass(frozen=True, slots=True)
class SceneBox:
    box_id: str
    center: Vector3
    size: Vector3
    color_role: int


@dataclass(slots=True)
class ObjectState:
    object_id: str
    position: Vector3
    size: Vector3
    mass_kg: float
    color_role: int
    attached: bool = False
    settled: bool = True


@dataclass(slots=True)
class RobotState:
    joints: JointVector
    command_azimuth: float
    command_reach: float
    command_height: float
    selected_axis: Coordinate = Coordinate.AZIMUTH
    gripper_open: bool = True
    gripper_aperture: float = 0.080
    joint_velocity_rad_s: JointVector = (0.0, 0.0, 0.0)
    joint_load_raw: tuple[int, int, int, int] = (0, 0, 0, 0)
    torque_enabled: tuple[bool, bool, bool, bool] = (True, True, True, True)
    supply_centivolts: int = 1200
    contact_load: float = 0.0
    last_motion_rejected: bool = False
    rejection_reason: str = ""

    def command(self) -> tuple[float, float, float]:
        return self.command_azimuth, self.command_reach, self.command_height


@dataclass(frozen=True, slots=True)
class MechanicsEvent:
    turn: int
    kind: str
    detail: str


@dataclass(slots=True)
class WorldState:
    robot: RobotState
    object: ObjectState
    barrier: SceneBox
    target: SceneBox
    attached_offset: Vector3 | None = None
    action_count: int = 0
    simulation_time_s: float = 0.0
    last_step_duration_s: float = 0.0
    success: bool = False
    level_failed: bool = False
    last_action: int = 0
    last_events: tuple[MechanicsEvent, ...] = ()
    event_log: list[MechanicsEvent] = field(default_factory=list)

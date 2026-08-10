"""Deterministic operational mechanics for the canonical live-run scene."""

from __future__ import annotations

import copy
from math import ceil, cos, pi, sin, sqrt
from numbers import Integral

from .config import (
    BARRIER_AZIMUTH_RAD,
    BARRIER_REACH_M,
    BARRIER_SIZE_M,
    BASE_HEIGHT_M,
    BASE_RADIUS_M,
    CANONICAL_SCENE_ID,
    COMMAND_AZIMUTH_LIMITS_RAD,
    COMMAND_HEIGHT_LIMITS_M,
    COMMAND_REACH_LIMITS_M,
    ELBOW_JOINT_RADIUS_M,
    GRIPPER_CLOSED_APERTURE_M,
    GRIPPER_JAW_DEPTH_M,
    GRIPPER_OPEN_APERTURE_M,
    GRIPPER_VERTICAL_TOLERANCE_M,
    INITIAL_AZIMUTH_RAD,
    INITIAL_HEIGHT_M,
    INITIAL_REACH_M,
    OBJECT_MASS_KG,
    OBJECT_SIZE_M,
    OBJECT_START_AZIMUTH_RAD,
    OBJECT_START_REACH_M,
    RATED_PAYLOAD_MOMENT_NM,
    ROUND1_ACTION_BUDGET,
    SCENE_SCHEMA_VERSION,
    SERVO_ACCELERATION_RAD_S2,
    SERVO_NO_LOAD_MAX_RAD_S,
    SERVO_SETTLING_TIME_S,
    SWEEP_MAX_JOINT_DELTA_RAD,
    TABLE_Z_M,
    TARGET_AZIMUTH_RAD,
    TARGET_REACH_M,
    TARGET_SIZE_M,
    WRIST_JOINT_RADIUS_M,
)
from .geometry import (
    boxes_overlap,
    capsules_overlap,
    inside_horizontal_target,
    point_segment_distance,
    segment_intersects_box,
    segment_intersects_vertical_cylinder,
    segment_intersects_yaw_box,
    sphere_intersects_box,
    sphere_intersects_vertical_cylinder,
    yaw_box_intersects_box,
    yaw_box_intersects_vertical_cylinder,
)
from .hardware import quantize_joints
from .interface import (
    ACTIONS,
    AZIMUTH_STEP_DEG,
    CONTROL_TURN_S,
    HEIGHT_STEP_M,
    REACH_STEP_M,
    Action,
    Coordinate,
)
from .kinematics import (
    ArmAnchors,
    cylindrical_from_tcp,
    exact_anchors,
    interpolate_joints,
    solve_cylindrical,
)
from .physical_geometry import (
    attached_world_position,
    attachment_local_offset,
    barrier_cap,
    gripper_boxes,
    robot_capsules,
    target_walls,
    workcell_solids,
)
from .world_state import (
    MechanicsEvent,
    ObjectState,
    RobotState,
    SceneBox,
    WorldState,
)


def polar_point(azimuth: float, reach: float, z: float) -> tuple[float, float, float]:
    from .config import BASE_AXIS_X_OFFSET_M

    return (
        BASE_AXIS_X_OFFSET_M + reach * cos(azimuth),
        reach * sin(azimuth),
        z,
    )


def canonical_state() -> WorldState:
    solution = solve_cylindrical(
        INITIAL_AZIMUTH_RAD,
        INITIAL_REACH_M,
        INITIAL_HEIGHT_M,
    )
    if solution is None:
        raise AssertionError("canonical initial command is unreachable")

    object_position = polar_point(
        OBJECT_START_AZIMUTH_RAD,
        OBJECT_START_REACH_M,
        TABLE_Z_M + OBJECT_SIZE_M[2] * 0.5,
    )
    barrier = SceneBox(
        box_id="barrier",
        center=polar_point(
            BARRIER_AZIMUTH_RAD,
            BARRIER_REACH_M,
            TABLE_Z_M + BARRIER_SIZE_M[2] * 0.5,
        ),
        size=BARRIER_SIZE_M,
        color_role=11,
    )
    target = SceneBox(
        box_id="target-bin",
        center=polar_point(
            TARGET_AZIMUTH_RAD,
            TARGET_REACH_M,
            TABLE_Z_M + TARGET_SIZE_M[2] * 0.5,
        ),
        size=TARGET_SIZE_M,
        color_role=8,
    )
    return WorldState(
        robot=RobotState(
            joints=quantize_joints(solution.joints),
            command_azimuth=INITIAL_AZIMUTH_RAD,
            command_reach=INITIAL_REACH_M,
            command_height=INITIAL_HEIGHT_M,
            gripper_aperture=GRIPPER_OPEN_APERTURE_M,
        ),
        object=ObjectState(
            object_id="workpiece",
            position=object_position,
            size=OBJECT_SIZE_M,
            mass_kg=OBJECT_MASS_KG,
            color_role=9,
        ),
        barrier=barrier,
        target=target,
    )


class OperationalWorld:
    """One deterministic world used by Python and browser parity fixtures."""

    def __init__(self, seed: int = 0) -> None:
        if isinstance(seed, bool) or not isinstance(seed, Integral):
            raise TypeError("seed must be an integer")
        self.seed = int(seed)
        self.state = canonical_state()
        self._update_servo_feedback()

    def reset(self) -> None:
        self.state = canonical_state()
        self._update_servo_feedback()

    def clone(self) -> "OperationalWorld":
        return copy.deepcopy(self)

    def current_collision_reason(self) -> str:
        """Return any interpenetration in the accepted authoritative pose."""

        return self._configuration_collision(
            exact_anchors(self.state.robot.joints),
            self.state.robot.gripper_aperture,
        )

    def _event(self, kind: str, detail: str) -> MechanicsEvent:
        return MechanicsEvent(self.state.action_count, kind, detail)

    def _finish_turn(self, action: int, events: list[MechanicsEvent]) -> None:
        self.state.last_action = action
        self.state.simulation_time_s += self.state.last_step_duration_s
        self.state.last_events = tuple(events)
        self.state.event_log.extend(events)

    def step(self, action: int) -> None:
        if isinstance(action, bool) or not isinstance(action, Integral):
            raise TypeError("action must be an integer")
        action_id = int(action)
        if action_id not in ACTIONS:
            raise ValueError(f"invalid action {action_id}; expected one of {ACTIONS}")
        if self.state.success or self.state.level_failed:
            raise RuntimeError("cannot step a terminal operational scene")

        self.state.action_count += 1
        self.state.last_step_duration_s = CONTROL_TURN_S
        self.state.robot.last_motion_rejected = False
        self.state.robot.rejection_reason = ""
        self.state.robot.contact_load = 0.0
        self.state.robot.joint_velocity_rad_s = (0.0, 0.0, 0.0)
        events: list[MechanicsEvent] = []
        selected = self.state.robot.selected_axis

        if action_id in (int(Action.PREVIOUS_COORDINATE), int(Action.NEXT_COORDINATE)):
            direction = -1 if action_id == int(Action.PREVIOUS_COORDINATE) else 1
            next_index = (int(selected) + direction) % len(Coordinate)
            self.state.robot.selected_axis = Coordinate(next_index)
            events.append(
                self._event(
                    "axis_selected",
                    self.state.robot.selected_axis.name.lower(),
                )
            )
        elif action_id == int(Action.OPEN_GRIPPER):
            events.extend(self._open_gripper())
        elif action_id == int(Action.CLOSE_GRIPPER):
            events.extend(self._close_gripper())
        else:
            direction = -1.0 if action_id == int(Action.DECREASE) else 1.0
            events.extend(self._coordinate_motion(direction))

        if (
            not self.state.success
            and self.state.action_count >= ROUND1_ACTION_BUDGET
        ):
            self.state.level_failed = True
            events.append(
                self._event(
                    "action_budget_exhausted",
                    f"budget={ROUND1_ACTION_BUDGET}",
                )
            )
        self._update_servo_feedback()
        self._finish_turn(action_id, events)

    def _coordinate_motion(self, direction: float) -> list[MechanicsEvent]:
        robot = self.state.robot
        azimuth, reach, height = robot.command()
        if robot.selected_axis is Coordinate.AZIMUTH:
            azimuth += direction * AZIMUTH_STEP_DEG * pi / 180.0
        elif robot.selected_axis is Coordinate.REACH:
            reach += direction * REACH_STEP_M
        else:
            height += direction * HEIGHT_STEP_M

        if not (
            COMMAND_AZIMUTH_LIMITS_RAD[0] <= azimuth <= COMMAND_AZIMUTH_LIMITS_RAD[1]
            and COMMAND_REACH_LIMITS_M[0] <= reach <= COMMAND_REACH_LIMITS_M[1]
            and COMMAND_HEIGHT_LIMITS_M[0] - 1e-9
            <= height
            <= COMMAND_HEIGHT_LIMITS_M[1]
        ):
            return [self._reject("command_bounds")]

        solution = solve_cylindrical(azimuth, reach, height)
        if solution is None:
            return [self._reject("inverse_kinematics")]

        candidate_joints = quantize_joints(solution.joints)
        legal, reason, final_anchors = self._swept_legality(candidate_joints)
        if not legal:
            return [self._reject(reason)]

        prior_joints = robot.joints
        prior_tcp = exact_anchors(robot.joints).tcp
        duration = self._motion_duration(prior_joints, candidate_joints)
        robot.joints = candidate_joints
        robot.joint_velocity_rad_s = tuple(
            (new - old) / duration
            for old, new in zip(prior_joints, candidate_joints, strict=True)
        )
        self.state.last_step_duration_s = duration
        robot.command_azimuth = azimuth
        robot.command_reach = reach
        robot.command_height = height
        if self.state.object.attached:
            offset = self.state.attached_offset
            if offset is None:
                raise AssertionError("attached object is missing its relative offset")
            final_yaw = cylindrical_from_tcp(final_anchors.tcp)[0]
            self.state.object.position = attached_world_position(
                final_anchors.tcp,
                final_yaw,
                offset,
            )
            self.state.object.settled = False
            robot.contact_load = self._normalized_load()
        displacement = sum(
            (new - old) ** 2 for new, old in zip(final_anchors.tcp, prior_tcp)
        ) ** 0.5
        return [
            self._event(
                "motion_accepted",
                f"{robot.selected_axis.name.lower()} displacement={displacement:.6f}m",
            )
        ]

    @staticmethod
    def _motion_duration(
        start: tuple[float, float, float],
        end: tuple[float, float, float],
    ) -> float:
        """Conservative synchronized bus-servo move time.

        Each joint follows a trapezoidal velocity profile.  The slowest joint
        determines completion, then a small settling allowance is added.  An
        action boundary never advances faster than the 250 ms host cadence.
        """

        longest = 0.0
        acceleration = SERVO_ACCELERATION_RAD_S2
        maximum_speed = SERVO_NO_LOAD_MAX_RAD_S
        ramp_distance = maximum_speed * maximum_speed / acceleration
        for old, new in zip(start, end, strict=True):
            distance = abs(new - old)
            if distance <= ramp_distance:
                motion = 2.0 * sqrt(distance / acceleration) if distance else 0.0
            else:
                motion = (
                    2.0 * maximum_speed / acceleration
                    + (distance - ramp_distance) / maximum_speed
                )
            longest = max(longest, motion)
        return max(CONTROL_TURN_S, longest + SERVO_SETTLING_TIME_S)

    def _update_servo_feedback(self) -> None:
        """Update deterministic approximations of bus-servo load and voltage.

        These values intentionally remain coarse signed servo-load readings,
        like the physical T=1051 packet.  They are not contact-force truth.
        """

        robot = self.state.robot
        _, shoulder, elbow = robot.joints
        payload_scale = 1.0 if self.state.object.attached else 0.0
        shoulder_load = int(round(95.0 + 42.0 * abs(sin(shoulder)) + 70.0 * payload_scale))
        elbow_load = int(round(48.0 + 36.0 * abs(sin(elbow)) + 44.0 * payload_scale))
        base_load = int(round(8.0 + 16.0 * abs(robot.joint_velocity_rad_s[0])))
        hand_load = 185 if self.state.object.attached else (18 if not robot.gripper_open else 4)
        if robot.last_motion_rejected:
            base_load = min(1023, base_load + 80)
            shoulder_load = min(1023, shoulder_load + 120)
            elbow_load = min(1023, elbow_load + 90)
        robot.joint_load_raw = (
            base_load,
            shoulder_load,
            elbow_load,
            hand_load,
        )
        sag = min(18, sum(abs(value) for value in robot.joint_load_raw) // 80)
        robot.supply_centivolts = 1200 - sag

    def _reject(self, reason: str) -> MechanicsEvent:
        robot = self.state.robot
        robot.last_motion_rejected = True
        robot.rejection_reason = reason
        robot.contact_load = 1.0
        return self._event("motion_rejected", reason)

    def _swept_legality(
        self,
        candidate_joints: tuple[float, float, float],
    ) -> tuple[bool, str, ArmAnchors]:
        final_anchors = exact_anchors(candidate_joints)
        for joints in interpolate_joints(
            self.state.robot.joints,
            candidate_joints,
            SWEEP_MAX_JOINT_DELTA_RAD,
        ):
            anchors = exact_anchors(joints)
            reason = self._configuration_collision(
                anchors,
                self.state.robot.gripper_aperture,
            )
            if reason:
                return False, reason, final_anchors
        return True, "", final_anchors

    def _configuration_collision(
        self,
        anchors: ArmAnchors,
        aperture: float,
    ) -> str:
        yaw = cylindrical_from_tcp(anchors.tcp)[0]
        column, upper, forearm, wrist = robot_capsules(anchors)
        moving_capsules = (upper, forearm, wrist)
        moving_joints = (
            (anchors.elbow, ELBOW_JOINT_RADIUS_M),
            (anchors.wrist, WRIST_JOINT_RADIUS_M),
        )
        gripper = gripper_boxes(anchors.tcp, yaw, aperture)

        object_center: tuple[float, float, float] | None = None
        object_box: SceneBox | None = None
        if self.state.object.attached:
            offset = self.state.attached_offset
            if offset is None:
                raise AssertionError("attached object is missing its relative offset")
            object_center = attached_world_position(anchors.tcp, yaw, offset)
            object_box = SceneBox(
                box_id=self.state.object.object_id,
                center=object_center,
                size=self.state.object.size,
                color_role=self.state.object.color_role,
            )

        if object_center is not None and (
            object_center[2] - self.state.object.size[2] * 0.5
            < TABLE_Z_M - 1.0e-8
        ):
            return "held_object_table_collision"
        for capsule in moving_capsules:
            if min(capsule.start[2], capsule.end[2]) - capsule.radius < TABLE_Z_M:
                return "arm_table_collision"
        for center, radius in moving_joints:
            if center[2] - radius < TABLE_Z_M:
                return "arm_table_collision"
        for body in gripper:
            if body.center[2] - body.size[2] * 0.5 < TABLE_Z_M:
                return "gripper_table_collision"

        if object_box is not None and yaw_box_intersects_vertical_cylinder(
            object_box.center,
            object_box.size,
            0.0,
            center_x=0.0,
            center_y=0.0,
            cylinder_radius=BASE_RADIUS_M,
            bottom=TABLE_Z_M,
            top=BASE_HEIGHT_M,
        ):
            return "held_object_base_collision"
        for capsule in moving_capsules:
            if segment_intersects_vertical_cylinder(
                capsule.start,
                capsule.end,
                capsule.radius,
                center_x=0.0,
                center_y=0.0,
                cylinder_radius=BASE_RADIUS_M,
                bottom=TABLE_Z_M,
                top=BASE_HEIGHT_M,
            ):
                return "arm_base_collision"
        for center, radius in moving_joints:
            if sphere_intersects_vertical_cylinder(
                center,
                radius,
                center_x=0.0,
                center_y=0.0,
                cylinder_radius=BASE_RADIUS_M,
                bottom=TABLE_Z_M,
                top=BASE_HEIGHT_M,
            ):
                return "arm_base_collision"
        for body in gripper:
            if yaw_box_intersects_vertical_cylinder(
                body.center,
                body.size,
                body.yaw,
                center_x=0.0,
                center_y=0.0,
                cylinder_radius=BASE_RADIUS_M,
                bottom=TABLE_Z_M,
                top=BASE_HEIGHT_M,
            ):
                return "gripper_base_collision"

        if (
            capsules_overlap(
                column.start,
                column.end,
                column.radius,
                forearm.start,
                forearm.end,
                forearm.radius,
            )
            or capsules_overlap(
                column.start,
                column.end,
                column.radius,
                wrist.start,
                wrist.end,
                wrist.radius,
            )
            or capsules_overlap(
                upper.start,
                upper.end,
                upper.radius,
                wrist.start,
                wrist.end,
                wrist.radius,
            )
            or point_segment_distance(
                anchors.wrist,
                column.start,
                column.end,
            )
            < WRIST_JOINT_RADIUS_M + column.radius
            or point_segment_distance(
                anchors.wrist,
                upper.start,
                upper.end,
            )
            < WRIST_JOINT_RADIUS_M + upper.radius
        ):
            return "arm_self_collision"
        for body in gripper:
            if (
                segment_intersects_yaw_box(
                    column.start,
                    column.end,
                    column.radius,
                    body.center,
                    body.size,
                    body.yaw,
                )
                or segment_intersects_yaw_box(
                    upper.start,
                    upper.end,
                    upper.radius,
                    body.center,
                    body.size,
                    body.yaw,
                )
            ):
                return "gripper_self_collision"
        if object_box is not None:
            if (
                segment_intersects_box(
                    column.start,
                    column.end,
                    object_box,
                    column.radius,
                )
                or segment_intersects_box(
                    upper.start,
                    upper.end,
                    object_box,
                    upper.radius,
                )
                or segment_intersects_box(
                    forearm.start,
                    forearm.end,
                    object_box,
                    forearm.radius,
                )
            ):
                return "held_object_self_collision"

        environment: tuple[tuple[str, SceneBox, float], ...] = (
            ("barrier", self.state.barrier, 0.002),
            ("barrier_cap", barrier_cap(self.state.barrier), 0.0),
            ("target", self.state.target, 0.0),
            *(
                ("target_wall", wall, 0.0)
                for wall in target_walls(self.state.target)
            ),
            *(
                ("workcell", solid, 0.0)
                for solid in workcell_solids()
            ),
        )
        for category, solid, object_margin in environment:
            if object_box is not None and boxes_overlap(
                object_box.center,
                object_box.size,
                solid,
                margin=object_margin,
            ):
                return f"held_object_{category}_collision"
            for capsule in moving_capsules:
                if segment_intersects_box(
                    capsule.start,
                    capsule.end,
                    solid,
                    capsule.radius,
                ):
                    return f"arm_{category}_collision"
            for center, radius in moving_joints:
                if sphere_intersects_box(center, radius, solid):
                    return f"arm_{category}_collision"
            for body in gripper:
                if yaw_box_intersects_box(
                    body.center,
                    body.size,
                    body.yaw,
                    solid,
                ):
                    return f"gripper_{category}_collision"
        return ""

    def _close_gripper(self) -> list[MechanicsEvent]:
        robot = self.state.robot
        if self.state.object.attached:
            robot.gripper_open = False
            robot.contact_load = self._normalized_load()
            return [self._event("gripper_closed", "object_already_attached")]

        tcp = exact_anchors(robot.joints).tcp
        obj = self.state.object
        delta_x = obj.position[0] - tcp[0]
        delta_y = obj.position[1] - tcp[1]
        delta_z = obj.position[2] - tcp[2]
        radial_axis = (cos(robot.command_azimuth), sin(robot.command_azimuth))
        jaw_axis = (-radial_axis[1], radial_axis[0])
        depth_offset = abs(delta_x * radial_axis[0] + delta_y * radial_axis[1])
        lateral_offset = abs(delta_x * jaw_axis[0] + delta_y * jaw_axis[1])
        object_width = max(obj.size[0], obj.size[1])

        bilateral = (
            depth_offset <= GRIPPER_JAW_DEPTH_M * 0.5
            and lateral_offset <= 0.006
            and abs(delta_z) <= GRIPPER_VERTICAL_TOLERANCE_M
            and object_width < GRIPPER_OPEN_APERTURE_M
        )
        target_aperture = (
            object_width if bilateral else GRIPPER_CLOSED_APERTURE_M
        )
        collision = self._aperture_sweep_collision(
            robot.gripper_aperture,
            target_aperture,
        )
        if collision:
            return [self._reject(collision)]

        robot.gripper_open = False
        if not bilateral:
            robot.gripper_aperture = target_aperture
            return [self._event("gripper_closed_empty", "bilateral_enclosure_failed")]

        robot.gripper_aperture = target_aperture
        robot.contact_load = 0.22
        obj.attached = True
        obj.settled = False
        self.state.attached_offset = attachment_local_offset(
            tcp,
            robot.command_azimuth,
            obj.position,
        )
        return [
            self._event("jaw_contact_left", obj.object_id),
            self._event("jaw_contact_right", obj.object_id),
            self._event("object_attached", obj.object_id),
        ]

    def _open_gripper(self) -> list[MechanicsEvent]:
        robot = self.state.robot
        collision = self._aperture_sweep_collision(
            robot.gripper_aperture,
            GRIPPER_OPEN_APERTURE_M,
        )
        if collision:
            return [self._reject(collision)]
        robot.gripper_open = True
        robot.gripper_aperture = GRIPPER_OPEN_APERTURE_M
        if not self.state.object.attached:
            return [self._event("gripper_opened", "empty")]

        obj = self.state.object
        obj.attached = False
        self.state.attached_offset = None
        support_id, support_top = self._highest_support_below(obj)
        settled_z = support_top + obj.size[2] * 0.5
        obj.position = (obj.position[0], obj.position[1], settled_z)
        obj.settled = True
        events = [
            self._event("object_released", obj.object_id),
            self._event(
                "gravity_settled",
                f"support={support_id} z={settled_z:.6f}",
            ),
        ]
        if (
            support_id == self.state.target.box_id
            and inside_horizontal_target(obj.position, obj.size, self.state.target)
        ):
            self.state.success = True
            events.append(self._event("level_completed", self.state.target.box_id))
        return events

    def _aperture_sweep_collision(
        self,
        start: float,
        end: float,
    ) -> str:
        anchors = exact_anchors(self.state.robot.joints)
        steps = max(1, int(ceil(abs(end - start) / 0.004)))
        for index in range(1, steps + 1):
            aperture = start + (end - start) * index / steps
            collision = self._configuration_collision(anchors, aperture)
            if collision:
                return collision
        return ""

    def _highest_support_below(self, obj: ObjectState) -> tuple[str, float]:
        """Return the highest stable support surface below the released object."""

        object_bottom = obj.position[2] - obj.size[2] * 0.5
        candidates: list[tuple[str, float]] = [("table", TABLE_Z_M)]
        for support in (self.state.barrier, self.state.target):
            support_top = support.center[2] + support.size[2] * 0.5
            if (
                support_top <= object_bottom + 1e-8
                and inside_horizontal_target(
                    obj.position,
                    obj.size,
                    support,
                    margin=0.0,
                )
            ):
                candidates.append((support.box_id, support_top))
        return max(candidates, key=lambda candidate: candidate[1])

    def _normalized_load(self) -> float:
        horizontal_reach = self.state.robot.command_reach
        moment = self.state.object.mass_kg * 9.80665 * horizontal_reach
        return min(1.0, moment / RATED_PAYLOAD_MOMENT_NM)

    def snapshot(self) -> dict[str, object]:
        state = self.state
        anchors = exact_anchors(state.robot.joints)
        return {
            "schemaVersion": SCENE_SCHEMA_VERSION,
            "sceneId": CANONICAL_SCENE_ID,
            "seed": self.seed,
            "turn": state.action_count,
            "simulationTimeS": round(state.simulation_time_s, 9),
            "robot": {
                "joints": list(state.robot.joints),
                "command": {
                    "azimuth": state.robot.command_azimuth,
                    "reach": state.robot.command_reach,
                    "height": state.robot.command_height,
                },
                "selectedAxis": state.robot.selected_axis.name.lower(),
                "gripperOpen": state.robot.gripper_open,
                "gripperAperture": state.robot.gripper_aperture,
                "contactLoad": state.robot.contact_load,
                "rejected": state.robot.last_motion_rejected,
                "rejectionReason": state.robot.rejection_reason,
                "anchors": {
                    "base": list(anchors.base),
                    "shoulder": list(anchors.shoulder),
                    "elbow": list(anchors.elbow),
                    "wrist": list(anchors.wrist),
                    "tcp": list(anchors.tcp),
                },
            },
            "object": {
                "id": state.object.object_id,
                "position": list(state.object.position),
                "size": list(state.object.size),
                "massKg": state.object.mass_kg,
                "attached": state.object.attached,
                "settled": state.object.settled,
            },
            "barrier": {
                "id": state.barrier.box_id,
                "center": list(state.barrier.center),
                "size": list(state.barrier.size),
            },
            "target": {
                "id": state.target.box_id,
                "center": list(state.target.center),
                "size": list(state.target.size),
            },
            "lastAction": state.last_action,
            "events": [
                {"turn": event.turn, "kind": event.kind, "detail": event.detail}
                for event in state.last_events
            ],
            "success": state.success,
            "terminal": state.success or state.level_failed,
        }


__all__ = ["OperationalWorld", "canonical_state", "polar_point"]

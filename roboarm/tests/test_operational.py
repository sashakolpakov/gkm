from __future__ import annotations

import copy
import json
from math import cos, sin

import numpy as np

from roboarm_game import Environment, make_env
from roboarm_game.canonical import CANONICAL_PICK_PLACE_ACTIONS
from roboarm_game.dynamics import OperationalWorld


def operational_env():
    return make_env("rb01-v1", seed=0, scenario="pick-place")


def test_operational_scene_conforms_to_public_protocol() -> None:
    env = operational_env()
    assert isinstance(env, Environment)
    frame = env.reset()
    assert frame.shape == (72, 128, 3)
    assert frame.dtype == np.uint8
    assert 0 <= int(frame.min()) <= int(frame.max()) <= 255
    assert len(np.unique(frame.reshape((-1, 3)), axis=0)) > 100
    telemetry = env.telemetry()
    assert telemetry["sensor_contract_id"] == "rb01-roarm-c920-v3"
    assert telemetry["sample"]["sequence"] == 0
    assert telemetry["arm"]["feedback"]["T"] == 1051
    assert env.actions == (1, 2, 3, 4, 5, 6)
    assert env.levels_completed == 0
    assert not env.terminal()


def test_operational_telemetry_is_closed_private_free_and_owned() -> None:
    env = operational_env()
    telemetry = env.telemetry()

    assert set(telemetry) == {
        "schema_version",
        "sensor_contract_id",
        "mode",
        "sample",
        "controller",
        "arm",
        "camera",
    }
    serialized = json.dumps(telemetry, sort_keys=True).lower()
    for forbidden in (
        "object_position",
        "target",
        "attached",
        "settled",
        "success",
        "terminal",
        "event",
        "safety",
        "verdict",
    ):
        assert forbidden not in serialized

    telemetry["controller"]["command_json"]["z"] = 999.0
    telemetry["arm"]["feedback"]["x"] = 999.0
    assert env.telemetry()["controller"]["command_json"]["z"] != 999.0
    assert env.telemetry()["arm"]["feedback"]["x"] != 999.0


def test_canonical_public_action_replay_completes_real_mechanics() -> None:
    env = operational_env()
    env.reset()
    event_kinds: list[str] = []
    for action in CANONICAL_PICK_PLACE_ACTIONS:
        env.step(action)
        event_kinds.extend(event["kind"] for event in env.snapshot()["events"])

    snapshot = env.snapshot()
    assert env.levels_completed == 1
    assert env.terminal()
    assert snapshot["success"] is True
    assert snapshot["object"]["attached"] is False
    assert snapshot["object"]["settled"] is True
    object_bottom = (
        snapshot["object"]["position"][2] - snapshot["object"]["size"][2] * 0.5
    )
    target_top = (
        snapshot["target"]["center"][2] + snapshot["target"]["size"][2] * 0.5
    )
    assert np.isclose(object_bottom, target_top, atol=1e-12, rtol=0.0)
    assert snapshot["turn"] == len(CANONICAL_PICK_PLACE_ACTIONS) == 63
    assert "jaw_contact_left" in event_kinds
    assert "jaw_contact_right" in event_kinds
    assert "object_attached" in event_kinds
    assert "object_released" in event_kinds
    assert "gravity_settled" in event_kinds
    assert event_kinds[-1] == "level_completed"


def test_target_floor_rejects_penetrating_carried_descent() -> None:
    env = operational_env()
    env.reset()
    for action in CANONICAL_PICK_PLACE_ACTIONS[:-3]:
        env.step(action)

    accepted_height = env.snapshot()["robot"]["command"]["height"]
    env.step(1)
    snapshot = env.snapshot()
    assert snapshot["robot"]["rejected"] is True
    assert snapshot["robot"]["rejectionReason"] == "held_object_target_collision"
    assert snapshot["robot"]["command"]["height"] == accepted_height
    assert snapshot["object"]["attached"] is True


def test_rejected_motion_changes_telemetry_without_faking_camera_pixels() -> None:
    env = operational_env()
    env.reset()
    for action in CANONICAL_PICK_PLACE_ACTIONS[:-3]:
        env.step(action)

    before_frame = env.frame()
    before_telemetry = env.telemetry()
    env.step(1)
    after_frame = env.frame()
    after_telemetry = env.telemetry()

    assert np.array_equal(after_frame, before_frame)
    assert after_telemetry["sample"]["sequence"] == before_telemetry["sample"]["sequence"] + 1
    assert after_telemetry["controller"]["interlocked"] is True
    assert after_telemetry["controller"]["command_json"] == before_telemetry["controller"]["command_json"]
    assert "reason" not in json.dumps(after_telemetry).lower()


def test_low_carry_is_rejected_by_full_gripper_swept_collision() -> None:
    env = operational_env()
    env.reset()
    approach_and_grasp = (4, 4, *([1] * 15), 6, *([2] * 7), 4)
    for action in approach_and_grasp:
        env.step(action)
    assert env.snapshot()["object"]["attached"] is True

    env.step(2)
    accepted_azimuth = env.snapshot()["robot"]["command"]["azimuth"]
    env.step(2)
    snapshot = env.snapshot()
    assert snapshot["robot"]["rejected"] is True
    assert snapshot["robot"]["rejectionReason"] == "gripper_barrier_collision"
    assert snapshot["robot"]["command"]["azimuth"] == accepted_azimuth
    assert snapshot["events"] == [
        {
            "turn": 28,
            "kind": "motion_rejected",
            "detail": "gripper_barrier_collision",
        }
    ]


def test_every_accepted_canonical_pose_is_free_of_rendered_solid_overlap() -> None:
    world = OperationalWorld(seed=0)
    assert world.current_collision_reason() == ""
    for action in CANONICAL_PICK_PLACE_ACTIONS:
        world.step(action)
        assert world.current_collision_reason() == ""


def test_obstacle_cap_and_target_walls_are_authoritative_colliders() -> None:
    cap_world = OperationalWorld(seed=0)
    cap_probe = (2, 4, *([1] * 4), 4, *([1] * 7))
    for action in cap_probe:
        cap_world.step(action)
    accepted_height = cap_world.state.robot.command_height
    cap_world.step(1)
    assert cap_world.state.robot.last_motion_rejected is True
    assert cap_world.state.robot.rejection_reason == (
        "gripper_barrier_cap_collision"
    )
    assert cap_world.state.robot.command_height == accepted_height

    wall_world = OperationalWorld(seed=0)
    wall_probe = (*([2] * 8), 4, *([1] * 4), 4, *([1] * 13))
    for action in wall_probe:
        wall_world.step(action)
    accepted_height = wall_world.state.robot.command_height
    wall_world.step(1)
    assert wall_world.state.robot.last_motion_rejected is True
    assert wall_world.state.robot.rejection_reason == (
        "gripper_target_wall_collision"
    )
    assert wall_world.state.robot.command_height == accepted_height


def test_attachment_offset_rotates_in_the_gripper_frame() -> None:
    world = OperationalWorld(seed=0)
    approach_lift_and_rotate = (
        4,
        2,
        4,
        *([1] * 15),
        6,
        *([2] * 14),
        4,
        *([2] * 4),
    )
    for action in approach_lift_and_rotate:
        world.step(action)

    assert world.state.object.attached is True
    tcp = world.snapshot()["robot"]["anchors"]["tcp"]
    object_position = world.state.object.position
    yaw = world.state.robot.command_azimuth
    delta_x = object_position[0] - tcp[0]
    delta_y = object_position[1] - tcp[1]
    radial_offset = delta_x * cos(yaw) + delta_y * sin(yaw)
    tangential_offset = -delta_x * sin(yaw) + delta_y * cos(yaw)
    assert np.isclose(radial_offset, -0.02, atol=4.0e-5, rtol=0.0)
    assert np.isclose(tangential_offset, 0.0, atol=4.0e-5, rtol=0.0)


def test_close_without_bilateral_enclosure_does_not_attach() -> None:
    env = operational_env()
    before = copy.deepcopy(env.snapshot()["object"])
    env.step(6)
    snapshot = env.snapshot()
    assert snapshot["object"] == before
    assert snapshot["object"]["attached"] is False
    assert snapshot["events"][0]["kind"] == "gripper_closed_empty"


def test_operational_clone_replay_and_divergence_are_exact() -> None:
    first = operational_env()
    for action in CANONICAL_PICK_PLACE_ACTIONS[:18]:
        first.step(action)
    second = first.clone()
    assert first.snapshot() == second.snapshot()
    assert np.array_equal(first.frame(), second.frame())

    for action in CANONICAL_PICK_PLACE_ACTIONS[18:32]:
        assert np.array_equal(first.step(action), second.step(action))
        assert first.snapshot() == second.snapshot()

    first_before = first.snapshot()
    second.step(3)
    assert first.snapshot() == first_before
    assert first.snapshot() != second.snapshot()


def test_snapshot_is_an_owned_copy_and_render_does_not_advance_state() -> None:
    env = operational_env()
    before = env.snapshot()
    frame = env.frame()
    frame[:, :, :] = 255
    mutated_snapshot = env.snapshot()
    mutated_snapshot["robot"]["command"]["height"] = 999.0
    assert env.snapshot() == before
    assert env.snapshot()["turn"] == 0
    assert not np.all(env.frame() == 255)


def test_unknown_scenario_rejected_without_fallback() -> None:
    try:
        make_env(scenario="arc")
    except ValueError as error:
        assert "unknown scenario" in str(error)
    else:
        raise AssertionError("unknown scenario unexpectedly constructed")

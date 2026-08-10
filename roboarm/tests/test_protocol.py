from __future__ import annotations

import numpy as np
import pytest

from roboarm_game import Environment, make_env
from roboarm_game.interface import (
    ACTIONS,
    AZIMUTH_STEP_DEG,
    HEIGHT_STEP_M,
    REACH_STEP_M,
)


def assert_frame_contract(frame: np.ndarray) -> None:
    assert frame.shape == (64, 64)
    assert frame.dtype == np.uint8
    assert int(frame.min()) >= 0
    assert int(frame.max()) <= 15


def test_factory_exposes_only_standalone_contract() -> None:
    env = make_env("rb01-v1", seed=0)
    assert isinstance(env, Environment)
    assert env.game_id == "rb01-v1"
    assert env.actions == (1, 2, 3, 4, 5, 6)
    assert env.actions is ACTIONS
    assert env.levels_completed == 0
    assert not env.terminal()
    assert_frame_contract(env.reset())


def test_interface_constants_match_normative_spec() -> None:
    assert AZIMUTH_STEP_DEG == 5.0
    assert REACH_STEP_M == 0.020
    assert HEIGHT_STEP_M == 0.015


def test_frame_and_reset_return_defensive_copies() -> None:
    env = make_env(seed=9)
    expected = env.reset()
    returned = env.frame()
    returned[:, :] = 15
    assert np.array_equal(env.frame(), expected)
    assert not np.shares_memory(returned, env.frame())

    stepped = env.step(2)
    stepped[:, :] = 0
    assert np.any(env.frame() != 0)

    reset = env.reset()
    fresh = make_env(seed=9).reset()
    assert np.array_equal(reset, fresh)


@pytest.mark.parametrize("bad_action", [0, 7, -1, 99])
def test_out_of_range_action_fails_before_mutation(bad_action: int) -> None:
    env = make_env()
    before = env.frame()
    with pytest.raises(ValueError, match="invalid action"):
        env.step(bad_action)
    assert np.array_equal(env.frame(), before)


@pytest.mark.parametrize("bad_action", [True, False, 1.0, "1", None])
def test_non_integer_action_fails_before_mutation(bad_action: object) -> None:
    env = make_env()
    before = env.frame()
    with pytest.raises(TypeError, match="action must be an integer"):
        env.step(bad_action)  # type: ignore[arg-type]
    assert np.array_equal(env.frame(), before)


def test_unknown_identity_and_bad_seed_are_rejected() -> None:
    with pytest.raises(ValueError, match="unknown game_id"):
        make_env("rb01")
    with pytest.raises(TypeError, match="seed must be an integer"):
        make_env(seed=True)


def test_every_documented_action_is_visibly_consumed() -> None:
    env = make_env()
    previous = env.reset()
    for action in (2, 1, 4, 3, 6, 5):
        current = env.step(action)
        assert_frame_contract(current)
        assert not np.array_equal(current, previous), f"action {action} was invisible"
        previous = current
    assert env.levels_completed == 0
    assert not env.terminal()


def test_command_limit_rejection_is_visible_and_deterministic() -> None:
    env = make_env()
    for _ in range(5):
        accepted = env.step(2)
    rejected = env.step(2)

    # The command bar is unchanged; status/action/turn telemetry changes.
    assert np.array_equal(accepted[5:10, 52:63], rejected[5:10, 52:63])
    assert np.all(rejected[43:46, 52:62] == 14)
    assert not np.array_equal(accepted, rejected)


def test_same_seed_and_actions_produce_identical_frames() -> None:
    sequence = (2, 2, 4, 1, 4, 2, 6, 5, 3)
    first = make_env(seed=781)
    second = make_env(seed=781)
    assert np.array_equal(first.reset(), second.reset())
    for action in sequence:
        assert np.array_equal(first.step(action), second.step(action))

"""Standalone rb01-v1 environment facade."""

from __future__ import annotations

import copy
from numbers import Integral

import numpy as np
from numpy.typing import NDArray

from .interface import ACTIONS, Action, COORDINATES
from .observation import calibration_telemetry
from .protocol import Environment
from .render import render_calibration
from .state import COMMAND_TICK_LIMIT, CalibrationState

Frame = NDArray[np.uint8]
GAME_ID = "rb01-v1"


class RoboArmEnv:
    """Deterministic unscored calibration shell for the Phase-0 protocol.

    No kinematics, collision, object mechanics, or scored levels are claimed by
    this class yet.  It exists to validate the apparatus contract in isolation.
    """

    def __init__(self, seed: int = 0) -> None:
        if isinstance(seed, bool) or not isinstance(seed, Integral):
            raise TypeError("seed must be an integer")
        self._seed = int(seed)
        self._state = CalibrationState()
        self._levels_completed = 0
        self._terminal = False
        self._frame = render_calibration(self._state)

    @property
    def game_id(self) -> str:
        return GAME_ID

    @property
    def actions(self) -> tuple[int, ...]:
        return ACTIONS

    @property
    def levels_completed(self) -> int:
        return self._levels_completed

    def terminal(self) -> bool:
        return self._terminal

    def reset(self) -> Frame:
        self._state = CalibrationState()
        self._levels_completed = 0
        self._terminal = False
        self._frame = render_calibration(self._state)
        return self.frame()

    def frame(self) -> Frame:
        return self._frame.copy()

    def telemetry(self) -> dict[str, object]:
        return copy.deepcopy(calibration_telemetry(self._state))

    def step(self, action: int) -> Frame:
        action_id = self._validated_action(action)
        if self._terminal:
            raise RuntimeError("cannot step a terminal environment")

        self._apply(Action(action_id))
        self._state.last_action = action_id
        self._state.turns += 1
        self._frame = render_calibration(self._state)
        return self.frame()

    def clone(self) -> "RoboArmEnv":
        return copy.deepcopy(self)

    @staticmethod
    def _validated_action(action: int) -> int:
        if isinstance(action, bool) or not isinstance(action, Integral):
            raise TypeError("action must be an integer")
        action_id = int(action)
        if action_id not in ACTIONS:
            raise ValueError(f"invalid action {action_id}; expected one of {ACTIONS}")
        return action_id

    def _apply(self, action: Action) -> None:
        self._state.rejected = False

        if action in (Action.DECREASE, Action.INCREASE):
            delta = -1 if action is Action.DECREASE else 1
            current = self._state.selected_ticks()
            candidate = current + delta
            if not -COMMAND_TICK_LIMIT <= candidate <= COMMAND_TICK_LIMIT:
                self._state.rejected = True
                return
            self._state.set_selected_ticks(candidate)
            return

        if action in (Action.PREVIOUS_COORDINATE, Action.NEXT_COORDINATE):
            direction = -1 if action is Action.PREVIOUS_COORDINATE else 1
            index = (int(self._state.selected) + direction) % len(COORDINATES)
            self._state.selected = COORDINATES[index]
            return

        if action is Action.OPEN_GRIPPER:
            self._state.gripper_open = True
            return

        if action is Action.CLOSE_GRIPPER:
            self._state.gripper_open = False
            return

        raise AssertionError(f"unhandled action {action}")


def make_env(
    game_id: str = GAME_ID,
    seed: int = 0,
    *,
    scenario: str = "calibration",
) -> Environment:
    """Construct a calibration or operational environment under one protocol."""

    if game_id != GAME_ID:
        raise ValueError(f"unknown game_id {game_id!r}; expected {GAME_ID!r}")
    if scenario == "calibration":
        return RoboArmEnv(seed=seed)
    if scenario in {"pick-place", "round-1"}:
        from .operational import OperationalRoboArmEnv

        return OperationalRoboArmEnv(seed=seed)
    raise ValueError(
        f"unknown scenario {scenario!r}; expected 'calibration', 'pick-place', "
        "or 'round-1'"
    )


__all__ = ["GAME_ID", "RoboArmEnv", "make_env"]

"""Protocol facade for the operational pick-and-place vertical slice."""

from __future__ import annotations

import copy
from numbers import Integral

import numpy as np
from numpy.typing import NDArray

from .dynamics import OperationalWorld
from .environment import GAME_ID
from .interface import ACTIONS
from .observation import operational_telemetry
from .render_world import render_operational

Frame = NDArray[np.uint8]


class OperationalRoboArmEnv:
    """Standalone environment backed by articulated operational mechanics."""

    def __init__(self, seed: int = 0) -> None:
        if isinstance(seed, bool) or not isinstance(seed, Integral):
            raise TypeError("seed must be an integer")
        self._seed = int(seed)
        self._world = OperationalWorld(self._seed)
        self._frame = render_operational(self._world.state)

    @property
    def game_id(self) -> str:
        return GAME_ID

    @property
    def actions(self) -> tuple[int, ...]:
        return ACTIONS

    @property
    def levels_completed(self) -> int:
        return int(self._world.state.success)

    def terminal(self) -> bool:
        return self._world.state.success or self._world.state.level_failed

    def reset(self) -> Frame:
        self._world.reset()
        self._frame = render_operational(self._world.state)
        return self.frame()

    def frame(self) -> Frame:
        return self._frame.copy()

    def telemetry(self) -> dict[str, object]:
        """Return only the public camera-companion controller packet."""

        return copy.deepcopy(operational_telemetry(self._world.state))

    def step(self, action: int) -> Frame:
        self._world.step(action)
        self._frame = render_operational(self._world.state)
        return self.frame()

    def clone(self) -> "OperationalRoboArmEnv":
        return copy.deepcopy(self)

    def snapshot(self) -> dict[str, object]:
        """Private validation/debug snapshot; excluded from the solver Protocol."""

        return copy.deepcopy(self._world.snapshot())


__all__ = ["OperationalRoboArmEnv"]

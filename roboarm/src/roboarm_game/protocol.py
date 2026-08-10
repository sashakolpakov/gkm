"""Public structural contract for ARC-style experimental environments.

This protocol is owned by ``roboarm_game``.  It intentionally does not inherit
from, emulate, or register with any external game or ARC API.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

Frame = NDArray[np.uint8]


@runtime_checkable
class Environment(Protocol):
    """The complete public runtime surface used by an rb01 solver."""

    @property
    def game_id(self) -> str:
        """Return the immutable versioned environment identity."""

    @property
    def actions(self) -> tuple[int, ...]:
        """Return all legal integer actions."""

    @property
    def levels_completed(self) -> int:
        """Return the sparse cumulative level-completion signal."""

    def reset(self) -> Frame:
        """Reset to the seeded initial state and return an owned frame."""

    def frame(self) -> Frame:
        """Return an owned copy of the current observation."""

    def telemetry(self) -> dict[str, object]:
        """Return an owned public controller-sensor packet."""

    def step(self, action: int) -> Frame:
        """Apply one legal action and return an owned next observation."""

    def clone(self) -> "Environment":
        """Return an exact, independently mutable environment copy."""

    def terminal(self) -> bool:
        """Report whether the current episode has terminated."""

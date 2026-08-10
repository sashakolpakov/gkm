"""Private Phase-0 calibration state.

This is deliberately a small command-state model, not robot dynamics.
"""

from __future__ import annotations

from dataclasses import dataclass

from .interface import Coordinate


COMMAND_TICK_LIMIT = 5


@dataclass(slots=True)
class CalibrationState:
    """Complete mutable state of the unscored Phase-0 shell."""

    selected: Coordinate = Coordinate.AZIMUTH
    azimuth_ticks: int = 0
    reach_ticks: int = 0
    height_ticks: int = 0
    gripper_open: bool = True
    rejected: bool = False
    last_action: int = 0
    turns: int = 0

    def selected_ticks(self) -> int:
        if self.selected is Coordinate.AZIMUTH:
            return self.azimuth_ticks
        if self.selected is Coordinate.REACH:
            return self.reach_ticks
        return self.height_ticks

    def set_selected_ticks(self, value: int) -> None:
        if self.selected is Coordinate.AZIMUTH:
            self.azimuth_ticks = value
        elif self.selected is Coordinate.REACH:
            self.reach_ticks = value
        else:
            self.height_ticks = value

    def command_ticks(self) -> tuple[int, int, int]:
        return self.azimuth_ticks, self.reach_ticks, self.height_ticks

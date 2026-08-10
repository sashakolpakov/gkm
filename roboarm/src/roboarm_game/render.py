"""Deterministic indexed-color renderer for the Phase-0 calibration shell."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .interface import Coordinate
from .state import COMMAND_TICK_LIMIT, CalibrationState

Frame = NDArray[np.uint8]
FRAME_SHAPE = (64, 64)
VIEWPORT_END = 50
PALETTE = {
    "background": 0,
    "table": 1,
    "structure": 2,
    "robot_base": 3,
    "upper_arm": 4,
    "forearm": 5,
    "gripper": 6,
    "target_border": 7,
    "target_interior": 8,
    "primary_object": 9,
    "secondary_object": 10,
    "obstacle": 11,
    "selected_axis": 12,
    "accepted_control": 13,
    "contact_or_rejected": 14,
    "success": 15,
}


def _line(
    frame: Frame,
    start: tuple[int, int],
    end: tuple[int, int],
    color: int,
) -> None:
    """Draw an integer Bresenham line with clipped endpoints."""

    x0, y0 = start
    x1, y1 = end
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    error = dx + dy
    while True:
        if 0 <= x0 < VIEWPORT_END and 0 <= y0 < FRAME_SHAPE[0]:
            frame[y0, x0] = color
        if x0 == x1 and y0 == y1:
            return
        doubled = 2 * error
        if doubled >= dy:
            error += dy
            x0 += sx
        if doubled <= dx:
            error += dx
            y0 += sy


def _draw_viewport(frame: Frame, state: CalibrationState) -> None:
    frame[36:60, 2:48] = PALETTE["table"]
    frame[35:37, 2:48] = PALETTE["structure"]
    frame[58:60, 2:48] = PALETTE["structure"]
    frame[36:60, 2:4] = PALETTE["structure"]
    frame[36:60, 46:48] = PALETTE["structure"]

    base = (11, 48)
    frame[46:54, 7:16] = PALETTE["robot_base"]
    frame[44:47, 9:14] = PALETTE["structure"]

    azimuth, reach, height = state.command_ticks()
    elbow = (22 + azimuth, 35 - height)
    target = (34 + 2 * reach + azimuth, 27 - 2 * height)
    target = (
        max(18, min(46, target[0])),
        max(8, min(34, target[1])),
    )

    _line(frame, base, elbow, PALETTE["upper_arm"])
    _line(frame, (elbow[0] + 1, elbow[1]), target, PALETTE["forearm"])
    frame[elbow[1] - 1 : elbow[1] + 2, elbow[0] - 1 : elbow[0] + 2] = (
        PALETTE["structure"]
    )

    tx, ty = target
    frame[ty - 1 : ty + 2, tx - 1 : tx + 2] = PALETTE["gripper"]
    jaw_gap = 3 if state.gripper_open else 1
    for offset in (-jaw_gap, jaw_gap):
        x = tx + offset
        if 0 <= x < VIEWPORT_END:
            frame[ty : min(ty + 4, 64), x] = PALETTE["gripper"]


def _draw_bar(
    frame: Frame,
    row: int,
    ticks: int,
    selected: bool,
) -> None:
    left = 52
    right = 62
    center = 57
    frame[row : row + 3, left : right + 1] = PALETTE["structure"]
    frame[row + 1, left + 1 : right] = PALETTE["background"]
    end = center + ticks
    lo, hi = sorted((center, end))
    frame[row + 1, lo : hi + 1] = PALETTE["accepted_control"]
    frame[row + 1, center] = PALETTE["gripper"]
    if selected:
        frame[row - 1 : row + 4, 50:52] = PALETTE["selected_axis"]


def _draw_hud(frame: Frame, state: CalibrationState) -> None:
    frame[:, 50] = PALETTE["structure"]
    frame[:, 63] = PALETTE["structure"]

    for coordinate, row, ticks in (
        (Coordinate.AZIMUTH, 6, state.azimuth_ticks),
        (Coordinate.REACH, 16, state.reach_ticks),
        (Coordinate.HEIGHT, 26, state.height_ticks),
    ):
        _draw_bar(frame, row, ticks, state.selected is coordinate)

    # Three stable one-pixel axis glyphs: diagonal, horizontal, vertical.
    frame[2, 53] = PALETTE["upper_arm"]
    frame[3, 54] = PALETTE["upper_arm"]
    frame[4, 55] = PALETTE["upper_arm"]
    frame[12, 53:57] = PALETTE["forearm"]
    frame[21:25, 55] = PALETTE["gripper"]

    # Wide jaws mean open; adjacent jaws mean closed.
    frame[34:39, 57] = PALETTE["gripper"]
    gap = 3 if state.gripper_open else 1
    frame[35:39, 57 - gap] = PALETTE["gripper"]
    frame[35:39, 57 + gap] = PALETTE["gripper"]

    status_color = (
        PALETTE["contact_or_rejected"]
        if state.rejected
        else PALETTE["accepted_control"]
    )
    frame[43:46, 52:62] = status_color

    frame[50:53, 52:62] = PALETTE["structure"]
    for action in range(1, 7):
        column = 52 + (action - 1) * 2
        frame[51, column] = (
            PALETTE["selected_axis"]
            if action == state.last_action
            else PALETTE["background"]
        )

    # A small five-bit turn counter makes every consumed action observable.
    frame[57:60, 52:62] = PALETTE["structure"]
    for bit in range(5):
        if state.turns & (1 << bit):
            frame[58, 52 + 2 * bit] = PALETTE["accepted_control"]


def render_calibration(state: CalibrationState) -> Frame:
    """Return a new 64×64 palette frame for the command-state shell."""

    frame = np.zeros(FRAME_SHAPE, dtype=np.uint8)
    _draw_viewport(frame, state)
    _draw_hud(frame, state)
    if frame.shape != FRAME_SHAPE or frame.dtype != np.uint8:
        raise AssertionError("renderer violated its frame contract")
    if int(frame.max()) > 15:
        raise AssertionError("renderer emitted a non-palette value")
    return frame


__all__ = ["FRAME_SHAPE", "VIEWPORT_END", "render_calibration"]

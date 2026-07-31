"""Trace avatar and competing courier during the optimized prefix."""

import gkm_try

from probe9_candidate import direct_second_prefix
from probe9_verify import ReachedLevel9, StopAtLevel9, boxes


def cell(frame, color):
    found = boxes(frame, color)
    if not found:
        return None
    row0, col0, row1, col1 = found[-1]
    return ((row0 + row1) // 8, (col0 + col1) // 8)


def inspect(env):
    try:
        gkm_try.resumed_solve(StopAtLevel9(env))
    except ReachedLevel9:
        pass

    path = direct_second_prefix()
    state = env.clone()
    for turn, action in enumerate(path, 1):
        state.step(action)
        if turn >= 20:
            avatar = cell(state.frame(), 14)
            thief = cell(state.frame(), 15)
            distance = None
            if avatar is not None and thief is not None:
                distance = (
                    abs(avatar[0] - thief[0])
                    + abs(avatar[1] - thief[1])
                )
            print(
                "THIEF_TRACE",
                turn,
                action,
                avatar,
                thief,
                distance,
                flush=True,
            )


gkm_try.A.run_program("wa30", inspect)

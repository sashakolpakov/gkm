"""Test two direct lower placements after the 28-turn double dismissal."""

import gkm_try

from probe8_entry import ReachedLevel8, StopAtLevel8
from probe8_reverse_idle import ROUTE
from probe8_reverse_stage import compact


FIRST_UNUSED = (
    [3, 3, 5, 4] + [1] * 3 + [4] * 8 + [2] * 3 + [5]
)
SECOND_UNUSED = (
    [1] * 3 + [3] * 11 + [2, 5, 1]
    + [4] * 10 + [2] * 2 + [5]
)


def inspect(env):
    try:
        gkm_try.resumed_solve(StopAtLevel8(env))
    except ReachedLevel8:
        pass
    clone = env.clone()
    base_level = clone.levels_completed
    actions = ROUTE + FIRST_UNUSED + SECOND_UNUSED
    prior = compact(clone, 0)
    for turn, action in enumerate(actions, 1):
        clone.step(action)
        current = compact(clone, turn)
        if (
            action == 5
            or current["empty"] != prior["empty"]
            or current["filled"] != prior["filled"]
            or current["level"] != base_level
        ):
            print("REVERSE_FINISH", action, current, flush=True)
        prior = current
        if clone.levels_completed > base_level or clone.terminal():
            break
    turn = len(actions)
    while clone.levels_completed == base_level and not clone.terminal():
        clone.step(5)
        turn += 1
        current = compact(clone, turn)
        if (
            current["empty"] != prior["empty"]
            or current["filled"] != prior["filled"]
            or current["level"] != base_level
        ):
            print("REVERSE_WAIT", current, flush=True)
        prior = current
    print("REVERSE_RESULT", compact(clone, turn), flush=True)


if __name__ == "__main__":
    gkm_try.A.run_program("wa30", inspect)

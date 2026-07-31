"""Compare compact finishes from the verified level-8 combined prefix."""

import gkm_try

from probe8_combo import REVERSE_TOP, UPPER_DELIVERY
from probe8_entry import ReachedLevel8, StopAtLevel8
from probe8_reverse_stage import compact


APPROACH_AND_DISMISS = (
    [3] * 7 + [2] * 3 + [4] * 5 + [2] * 4 + [5, 3, 5]
)
PREFIX = REVERSE_TOP + UPPER_DELIVERY + APPROACH_AND_DISMISS

TO_ROW13_RIGHT = (
    [1] + [3] * 4 + [2, 5, 1] + [4] * 10 + [2] * 2 + [5]
)
TO_ROW14_RIGHT = (
    [1] + [3] * 4 + [2, 5, 1] + [4] * 10 + [2] * 3 + [5]
)


def finish(base, suffix, label):
    clone = base.clone()
    base_level = clone.levels_completed
    turn = len(PREFIX)
    for action in suffix:
        clone.step(action)
        turn += 1
        if clone.levels_completed > base_level or clone.terminal():
            break
    placed = compact(clone, turn)
    print(label + "_PLACED", placed, flush=True)
    prior = (placed["empty"], placed["filled"])
    while clone.levels_completed == base_level and not clone.terminal():
        clone.step(5)
        turn += 1
        current = compact(clone, turn)
        progress = (current["empty"], current["filled"])
        if progress != prior:
            print(label + "_EVENT", current, flush=True)
        prior = progress
    print(label + "_RESULT", compact(clone, turn), flush=True)


def inspect(env):
    try:
        gkm_try.resumed_solve(StopAtLevel8(env))
    except ReachedLevel8:
        pass
    start = env.clone()
    for action in PREFIX:
        start.step(action)
    print("FINISH_START", compact(start, len(PREFIX)), flush=True)
    finish(start, (), "IDLE")
    finish(start, TO_ROW13_RIGHT, "ROW13")
    finish(start, TO_ROW14_RIGHT, "ROW14")


if __name__ == "__main__":
    gkm_try.A.run_program("wa30", inspect)

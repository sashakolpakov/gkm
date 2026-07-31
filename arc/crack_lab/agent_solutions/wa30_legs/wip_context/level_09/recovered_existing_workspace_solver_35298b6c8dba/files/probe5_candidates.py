"""Test faster direct-cargo choices on level 5."""

import gkm_try

from probe5_structure import PHASES, summary
from probe_minimize_segments import CaptureSegments


MID_DIRECT_ALIGN = [1, 1, 5, 2] + [3] * 8 + [5, 1, 4]


def run(start, phases, label):
    clone = start.clone()
    base_level = clone.levels_completed
    turn = 0
    for actions in phases:
        for action in actions:
            clone.step(action)
            turn += 1
            if clone.levels_completed > base_level or clone.terminal():
                break
        if clone.levels_completed > base_level or clone.terminal():
            break
    while clone.levels_completed == base_level and not clone.terminal():
        clone.step(5)
        turn += 1
    print(label, summary(clone, turn), flush=True)


def inspect(env):
    capture = CaptureSegments(env)
    gkm_try.resumed_solve(capture)
    start = capture.starts[4]
    tail = [actions for _, actions in PHASES[1:]]
    run(start, [MID_DIRECT_ALIGN] + tail, "MID_REPLACE")
    run(
        start,
        [MID_DIRECT_ALIGN, [2] + [4] * 7] + [actions for _, actions in PHASES],
        "MID_ADD",
    )


if __name__ == "__main__":
    gkm_try.A.run_program("wa30", inspect)

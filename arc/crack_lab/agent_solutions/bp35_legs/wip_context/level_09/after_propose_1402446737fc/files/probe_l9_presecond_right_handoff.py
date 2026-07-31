"""Leave the far-right descent at its last safe landing with a control."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from probe_l9_boosted_supported_landing import report
from probe_l9_control_row import controls
from probe_l9_presecond_right_flip import right_end
from probe_l9_route_deletions import enter_level_9
from probe_l9_right_trap_stage import full_catches


def landing(root, switch_index):
    child = right_end(root)
    child.step(*controls(child)[switch_index])
    child.step(6, 45, 33)
    child.step(6, 45, 33)
    return child


def probe(env):
    enter_level_9(env)
    switch_index = int(sys.argv[1])
    side = sys.argv[2]
    child = landing(env, switch_index)
    if side == "left":
        x, move = 39, 3
    else:
        x, move = 51, 4
        print("LANDING_FULL", full_catches(child), flush=True)
    child.step(6, x, 27)
    child.step(move)
    report((switch_index, side, "HANDOFF"), child)
    if side == "right":
        child.step(6, 45, 35)
        report((switch_index, side, "STAGE_C8"), child)
        child.step(6, 45, 41)
        report((switch_index, side, "STAGE_C8_BELOW"), child)
        child.step(6, 51, 35)
        report((switch_index, side, "DROP_C8"), child)
        print("C8_FULL", full_catches(child), flush=True)
        return
        child.step(6, 57, 27)
        report((switch_index, side, "OUTER_CLEAR"), child)
        if child.terminal():
            return
        child.step(4)
        x = 57
        report((switch_index, side, "OUTER"), child)
        if child.terminal():
            return
        print("FULL", full_catches(child), flush=True)
    for depth in range(1, 9):
        child.step(6, x, 35 if side == "right" else 33)
        report((switch_index, side, depth), child)
        if child.terminal() or int(child.levels_completed) >= 9:
            break


if __name__ == "__main__":
    arena.run_program("bp35", probe)

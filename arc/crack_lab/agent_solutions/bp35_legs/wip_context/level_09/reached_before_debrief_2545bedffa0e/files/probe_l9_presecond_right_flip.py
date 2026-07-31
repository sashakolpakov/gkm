"""Reverse at PRE_SECOND's far-right corridor while preserving one control."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from probe_l9_boosted_supported_landing import report
from probe_l9_control_row import controls
from probe_l9_presecond_outer import pre_second
from probe_l9_route_deletions import enter_level_9
from probe_l9_right_trap_stage import full_catches


def right_end(root):
    child = pre_second(root)
    for target in range(3, 8):
        x = 3 + 6 * target
        if int(child.frame()[39][x]) in (12, 14, 15):
            child.step(6, x, 39)
        child.step(4)
    return child


def probe(env):
    enter_level_9(env)
    child = right_end(env)
    report(("BEFORE_FLIP", "FULL", full_catches(child)), child)
    switch_index = int(sys.argv[1])
    switch = controls(child)[switch_index]
    child.step(*switch)
    report((switch_index, switch, "FLIP"), child)
    if child.terminal():
        return
    for depth in range(1, 9):
        child.step(6, 45, 33)
        report((switch_index, "DROP", depth), child)
        if child.terminal() or int(child.levels_completed) >= 9:
            break


if __name__ == "__main__":
    arena.run_program("bp35", probe)

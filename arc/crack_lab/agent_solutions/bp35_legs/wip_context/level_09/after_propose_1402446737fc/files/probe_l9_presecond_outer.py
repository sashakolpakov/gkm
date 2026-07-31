"""Climb from PRE_SECOND while retaining both controls and seek the outer opening."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from probe_l9_boosted_supported_landing import report
from probe_l9_control_row import controls
from probe_l9_route_deletions import enter_level_9
from probe_l9_super_final_exit import super_boosted


def pre_second(root):
    child = super_boosted(root)
    child.step(*controls(child)[0])
    child.step(*controls(child)[0])
    for action in ((6, 21, 39), 3, (6, 15, 39), 3):
        child.step(*action) if isinstance(action, tuple) else child.step(action)
    return child


def probe(env):
    enter_level_9(env)
    child = pre_second(env)
    for height in range(10):
        report((height, "LEFT_COLOR", int(child.frame()[39][9])), child)
        if child.terminal() or int(child.levels_completed) >= 9:
            break
        child.step(6, 15, 33)


if __name__ == "__main__":
    arena.run_program("bp35", probe)

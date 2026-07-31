"""Align with the lower wall gap before consuming the retained switch."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from probe_l9_control_row import controls
from probe_l9_route_deletions import enter_level_9
from probe_l9_skip4_switch_choices import report
from probe_l9_supported_high_path import high_high


DROP = (6, 27, 33)


def aligned(root, depth, column):
    child = high_high(root)
    child.step(4)
    for _ in range(depth):
        child.step(*DROP)
    for col in range(5, column + 1):
        child.step(6, 3 + 6 * col, 27)
        child.step(4)
    return child


def probe(env):
    enter_level_9(env)
    for depth in (4, 5, 6, 7):
        for column in (5, 6, 7):
            child = aligned(env, depth, column)
            report((depth, column, "ALIGNED"), child)
            visible = controls(child)
            if not visible or child.terminal():
                continue
            child.step(*visible[-1])
            report((depth, column, "FLIP"), child)
            for direction in (3, 4):
                branch = child.clone()
                for _ in range(6):
                    branch.step(direction)
                    if branch.terminal() or int(branch.levels_completed) >= 9:
                        break
                report((depth, column, direction), branch)


if __name__ == "__main__":
    arena.run_program("bp35", probe)

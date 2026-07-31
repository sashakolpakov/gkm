"""Reverse the retained switch near the lower barrier of the staged path."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from probe_l9_control_row import controls
from probe_l9_route_deletions import enter_level_9
from probe_l9_skip4_switch_choices import report
from probe_l9_supported_high_path import high_high


DROP = (6, 27, 33)


def c4(root, depth):
    child = high_high(root)
    child.step(4)
    for _ in range(depth):
        child.step(*DROP)
    return child


def probe(env):
    enter_level_9(env)
    for depth in range(5, 10):
        child = c4(env, depth)
        report((depth, "PRE"), child)
        visible = controls(child)
        if not visible or child.terminal():
            continue
        child.step(*visible[-1])
        report((depth, "FLIP"), child)
        for direction in (3, 4):
            branch = child.clone()
            for _ in range(8):
                branch.step(direction)
                if branch.terminal() or int(branch.levels_completed) >= 9:
                    break
            report((depth, direction, "MOVE8"), branch)


if __name__ == "__main__":
    arena.run_program("bp35", probe)

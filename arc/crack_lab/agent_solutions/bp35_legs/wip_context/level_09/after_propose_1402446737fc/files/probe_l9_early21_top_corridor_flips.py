"""Carry the surviving height-seven control across the top corridor before flipping."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from probe_l9_control_row import controls
from probe_l9_early21_c9_up import climbed
from probe_l9_early21_right import state
from probe_l9_route_deletions import enter_level_9


def top_column(root, target):
    child = climbed(root, 2)
    for col in range(8, target - 1, -1):
        child.step(6, 3 + 6 * col, 39)
        child.step(3)
    return child


def probe(env):
    enter_level_9(env)
    for target in range(9, 1, -1):
        child = top_column(env, target)
        state((target, "TOP"), child)
        switch = controls(child)[0]
        child.step(*switch)
        state((target, switch, "DOWN"), child)
        if child.terminal():
            continue
        for action in ((3,), (4,), (6, 3 + 6 * target, 35)):
            branch = child.clone()
            branch.step(*action)
            state((target, "ACTION", action), branch)


if __name__ == "__main__":
    arena.run_program("bp35", probe)

"""Use the reappearing third control in each upper-wall landing column."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from probe_l9_control_row import controls
from probe_l9_early21_right import state
from probe_l9_early21_top_corridor_flips import top_column
from probe_l9_route_deletions import enter_level_9


def down_column(root, target):
    child = top_column(root, target)
    child.step(*controls(child)[0])
    return child


def third_flipped(root, target):
    child = down_column(root, target)
    child.step(*controls(child)[0])
    return child


def probe(env):
    enter_level_9(env)
    for target in range(2, 9):
        child = down_column(env, target)
        state((target, "DOWN"), child)
        switch = controls(child)[0]
        child.step(*switch)
        state((target, switch, "UP"), child)
        if child.terminal():
            continue
        x = 3 + 6 * target
        for height in range(1, 6):
            child.step(6, x, 33)
            state((target, "CLIMB", height), child)
            if child.terminal() or int(child.levels_completed) >= 9:
                break


if __name__ == "__main__":
    arena.run_program("bp35", probe)

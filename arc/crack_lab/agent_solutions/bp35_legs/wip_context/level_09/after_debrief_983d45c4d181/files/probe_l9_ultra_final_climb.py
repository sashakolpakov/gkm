"""Climb the two new left shafts after the final retained reversal."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from probe_l9_boosted_supported_landing import report
from probe_l9_control_row import controls
from probe_l9_route_deletions import enter_level_9
from probe_l9_ultra_cycle_descent import arrested
from probe_l9_ultra_final_align import walk_left


def flipped(root, depth, column):
    child = arrested(root)
    for _ in range(depth):
        child.step(6, 27, 33)
    walk_left(child, column)
    child.step(*controls(child)[-1])
    return child


def probe(env):
    enter_level_9(env)
    depth = int(sys.argv[1])
    column = int(sys.argv[2])
    child = flipped(env, depth, column)
    x = 3 + 6 * column
    report((depth, column, 0), child)
    for height in range(1, 13):
        child.step(6, x, 33)
        report((depth, column, height), child)
        if child.terminal() or int(child.levels_completed) >= 9:
            break


if __name__ == "__main__":
    arena.run_program("bp35", probe)

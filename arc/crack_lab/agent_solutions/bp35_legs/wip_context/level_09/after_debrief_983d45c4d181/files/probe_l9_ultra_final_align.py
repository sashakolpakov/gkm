"""Screen the last retained reversal across the left gate and descent depths."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from probe_l9_boosted_supported_landing import report
from probe_l9_control_row import controls
from probe_l9_route_deletions import enter_level_9
from probe_l9_ultra_cycle_descent import arrested


def walk_left(child, target_column):
    for column in range(3, target_column - 1, -1):
        x = 3 + 6 * column
        if int(child.frame()[27][x]) not in (9, 10, 11):
            child.step(6, x, 27)
        child.step(3)


def probe(env):
    enter_level_9(env)
    depth = int(sys.argv[1])
    column = int(sys.argv[2])
    child = arrested(env)
    for _ in range(depth):
        child.step(6, 27, 33)
    walk_left(child, column)
    report((depth, column, "ALIGNED"), child)
    if child.terminal() or not controls(child):
        return
    switch = controls(child)[-1]
    child.step(*switch)
    report((depth, column, switch, "FLIP"), child)


if __name__ == "__main__":
    arena.run_program("bp35", probe)

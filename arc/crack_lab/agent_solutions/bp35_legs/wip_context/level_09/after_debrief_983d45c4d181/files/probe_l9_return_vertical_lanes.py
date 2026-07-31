"""Screen vertical exits across columns two through five in the returned room."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from probe_l9_route_deletions import enter_level_9
from probe_l9_skip4_switch_choices import report
from probe_l9_supported_gap_climb import flipped


def c2(root):
    child = flipped(root, 6)
    for action in (3, 3, 3, (6, 15, 39), 3):
        child.step(*action) if isinstance(action, tuple) else child.step(action)
    return child


def at_column(root, column):
    child = c2(root)
    for col in range(3, column + 1):
        child.step(6, 3 + 6 * col, 39)
        child.step(4)
    return child


def probe(env):
    enter_level_9(env)
    for column in range(2, 6):
        for staged in (False, True):
            child = at_column(env, column)
            report((column, staged, "ROOT"), child)
            x = 3 + 6 * column
            if staged:
                child.step(6, x, 45)
                report((column, staged, "BELOW"), child)
            child.step(6, x, 33)
            report((column, staged, "ABOVE"), child)


if __name__ == "__main__":
    arena.run_program("bp35", probe)

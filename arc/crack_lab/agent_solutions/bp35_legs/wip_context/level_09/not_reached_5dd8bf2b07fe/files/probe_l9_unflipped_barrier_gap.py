"""Leave column six through either side gap before the barrier becomes solid."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from probe_l9_route_deletions import enter_level_9
from probe_l9_skip4_switch_choices import report
from probe_l9_supported_final_alignment import aligned


def gap(root, column):
    child = aligned(root, 6, 6)
    for _ in range(4):
        child.step(6, 39, 33)
    x = 3 + 6 * column
    child.step(6, x, 27)
    child.step(3 if column < 6 else 4)
    return child


def probe(env):
    enter_level_9(env)
    for column in (5, 7):
        child = gap(env, column)
        report((column, 0), child)
        x = 3 + 6 * column
        for depth in range(1, 10):
            child.step(6, x, 33)
            report((column, depth), child)
            if child.terminal() or int(child.levels_completed) >= 9:
                break


if __name__ == "__main__":
    arena.run_program("bp35", probe)

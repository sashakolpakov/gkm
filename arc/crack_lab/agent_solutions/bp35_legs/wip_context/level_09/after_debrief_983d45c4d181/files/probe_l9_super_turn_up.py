"""Turn upward where a lower horizontal traverse meets its first wall."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from probe_l9_boosted_supported_landing import report
from probe_l9_route_deletions import enter_level_9
from probe_l9_super_horizontal import start


def probe(env):
    enter_level_9(env)
    depth, column, height = map(int, sys.argv[1:4])
    child = start(env, depth, column, height)
    current = column
    for target in range(column + 1, 9):
        x = 3 + 6 * target
        color = int(child.frame()[39][x])
        if color in (3, 5):
            break
        if color in (12, 14, 15):
            child.step(6, x, 39)
        child.step(4)
        current = target
    report((depth, column, height, current, "TURN"), child)
    x = 3 + 6 * current
    for climb in range(1, 10):
        child.step(6, x, 33)
        report((climb, x), child)
        if child.terminal() or int(child.levels_completed) >= 9:
            break


if __name__ == "__main__":
    arena.run_program("bp35", probe)

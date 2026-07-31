"""Test one lateral handoff from the first safe return-climb landing."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from probe_l9_boosted_supported_landing import report
from probe_l9_right_trap_climb import reversed_root
from probe_l9_route_deletions import enter_level_9


BRANCHES = {
    "left": ((6, 21, 39), 3),
    "right": ((6, 33, 39), 4),
    "left_below": ((6, 21, 45), (6, 21, 39), 3),
    "right_below": ((6, 33, 45), (6, 33, 39), 4),
}


def probe(env):
    enter_level_9(env)
    child = reversed_root(env)
    child.step(6, 27, 33)
    report((sys.argv[1], 0), child)
    for index, action in enumerate(BRANCHES[sys.argv[1]], 1):
        child.step(*action) if isinstance(action, tuple) else child.step(action)
        report((sys.argv[1], index, action), child)
        if child.terminal() or int(child.levels_completed) >= 9:
            break


if __name__ == "__main__":
    arena.run_program("bp35", probe)

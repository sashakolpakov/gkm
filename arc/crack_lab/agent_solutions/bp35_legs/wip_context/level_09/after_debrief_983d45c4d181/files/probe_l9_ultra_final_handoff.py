"""Test lateral exits at the highest safe final left-shaft landing."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from probe_l9_boosted_supported_landing import report
from probe_l9_route_deletions import enter_level_9
from probe_l9_ultra_final_climb import flipped


BRANCHES = {
    "left": ((6, 15, 39), 3),
    "right": ((6, 27, 39), 4),
    "left_above": ((6, 15, 33), (6, 15, 39), 3),
    "right_above": ((6, 27, 33), (6, 27, 39), 4),
    "left_below": ((6, 15, 45), (6, 15, 39), 3),
    "right_below": ((6, 27, 45), (6, 27, 39), 4),
}


def frontier(root):
    child = flipped(root, 3, 3)
    for _ in range(3):
        child.step(6, 21, 33)
    return child


def probe(env):
    enter_level_9(env)
    child = frontier(env)
    name = sys.argv[1]
    report((name, 0), child)
    for index, action in enumerate(BRANCHES[name], 1):
        child.step(*action) if isinstance(action, tuple) else child.step(action)
        report((name, index, action), child)
        if child.terminal() or int(child.levels_completed) >= 9:
            break


if __name__ == "__main__":
    arena.run_program("bp35", probe)

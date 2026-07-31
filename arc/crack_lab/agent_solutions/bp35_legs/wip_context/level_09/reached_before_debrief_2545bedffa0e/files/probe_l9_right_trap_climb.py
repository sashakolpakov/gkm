"""Climb the catch shaft above the safe boosted trapdoor reversal."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from probe_l9_boosted_supported_landing import report
from probe_l9_control_row import controls
from probe_l9_right_trap_flips import aligned
from probe_l9_route_deletions import enter_level_9


def reversed_root(root):
    child = aligned(root, 4, 4)
    child.step(*controls(child)[-1])
    return child


def probe(env):
    enter_level_9(env)
    child = reversed_root(env)
    report(0, child)
    for height in range(1, 13):
        child.step(6, 27, 33)
        report(height, child)
        if child.terminal() or int(child.levels_completed) >= 9:
            break


if __name__ == "__main__":
    arena.run_program("bp35", probe)

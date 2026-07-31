"""Traverse right from PRE_SECOND without consuming either retained control."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from probe_l9_boosted_supported_landing import report
from probe_l9_presecond_outer import pre_second
from probe_l9_route_deletions import enter_level_9


def probe(env):
    enter_level_9(env)
    child = pre_second(env)
    current = 2
    report((current, "START"), child)
    for target in range(3, 9):
        x = 3 + 6 * target
        color = int(child.frame()[39][x])
        print("TARGET", target, "color", color, flush=True)
        if color in (3, 5):
            break
        if color in (12, 14, 15):
            child.step(6, x, 39)
            report((target, "CLEAR"), child)
            if child.terminal():
                return
        child.step(4)
        current = target
        report((current, "RIGHT"), child)
        if child.terminal() or int(child.levels_completed) >= 9:
            return
    x = 3 + 6 * current
    for height in range(1, 8):
        child.step(6, x, 33)
        report((current, "CLIMB", height), child)
        if child.terminal() or int(child.levels_completed) >= 9:
            return


if __name__ == "__main__":
    arena.run_program("bp35", probe)

"""Trace controlled descent after the retained-switch second arrest."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from probe_l9_boosted_supported_landing import report
from probe_l9_control_row import controls
from probe_l9_route_deletions import enter_level_9
from probe_l9_ultra_cycle_stage import height_eight


def arrested(root):
    child = height_eight(root)
    child.step(6, 21, 45)
    child.step(*controls(child)[0])
    return child


def probe(env):
    enter_level_9(env)
    child = arrested(env)
    report(0, child)
    for depth in range(1, 8):
        child.step(6, 27, 33)
        report(depth, child)
        if child.terminal() or int(child.levels_completed) >= 9:
            break


if __name__ == "__main__":
    arena.run_program("bp35", probe)

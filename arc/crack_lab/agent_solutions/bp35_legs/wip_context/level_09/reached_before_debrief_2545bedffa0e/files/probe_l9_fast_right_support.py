"""Probe the supported right shaft before the left-trapdoor camera shift."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from probe_l9_boosted_supported_landing import report
from probe_l9_control_row import controls
from probe_l9_preserve_boost import boosted
from probe_l9_right_trap_stage import full_catches
from probe_l9_route_deletions import enter_level_9


def root_at_right(root):
    child = boosted(root)
    child.step(*controls(child)[0])
    child.step(3)
    return child


def probe(env):
    enter_level_9(env)
    child = root_at_right(env)
    report("ROOT", child)
    print("FULL", full_catches(child), flush=True)
    for action in ((6, 21, 33), (6, 21, 39), (6, 21, 41)):
        branch = child.clone()
        branch.step(*action)
        report(action, branch)

    descent = child.clone()
    descent.step(6, 21, 33)
    descent.step(6, 21, 35)
    report("YELLOW_LANDING", descent)
    for depth in range(1, 9):
        descent.step(6, 21, 33)
        report(("DESCENT", depth), descent)
        if descent.terminal() or int(descent.levels_completed) >= 9:
            break


if __name__ == "__main__":
    arena.run_program("bp35", probe)

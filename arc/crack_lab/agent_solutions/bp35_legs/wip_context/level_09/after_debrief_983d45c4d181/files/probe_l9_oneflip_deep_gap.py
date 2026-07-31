"""Walk from the aligned trapdoor landing into the deep partial-wall gap."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from probe_l9_boosted_supported_landing import report
from probe_l9_control_row import controls
from probe_l9_oneflip_lower import oneflip
from probe_l9_oneflip_trapdoors import yellows
from probe_l9_route_deletions import enter_level_9


def deep(root):
    child = oneflip(root, 2)
    left_yellow = min(yellows(child), key=lambda action: action[1])
    child.step(*left_yellow)
    child.step(*controls(child)[-1])
    return child


def probe(env):
    enter_level_9(env)
    child = deep(env)
    report("DEEP", child)
    for index in range(1, 9):
        child.step(4)
        report(("RIGHT", index), child)
        if child.terminal() or int(child.levels_completed) >= 9:
            break


if __name__ == "__main__":
    arena.run_program("bp35", probe)

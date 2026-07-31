"""Stage a catch under the occupied left yellow trapdoor before opening it."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from probe_l9_boosted_supported_landing import report
from probe_l9_control_row import controls
from probe_l9_oneflip_trapdoors import yellows
from probe_l9_route_deletions import enter_level_9
from probe_l9_super_final_exit import super_boosted


def landed(root):
    child = super_boosted(root)
    child.step(*controls(child)[0])
    child.step(3)
    child.step(3)
    child.step(6, 21, 39)
    child.step(*min(yellows(child), key=lambda action: action[1]))
    return child


def probe(env):
    enter_level_9(env)
    child = landed(env)
    report(0, child)
    for depth in range(1, 11):
        child.step(6, 15, 33)
        report(depth, child)
        if child.terminal() or int(child.levels_completed) >= 9:
            break


if __name__ == "__main__":
    arena.run_program("bp35", probe)

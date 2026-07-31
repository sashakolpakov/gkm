"""Align the row-35 retained control over the yellow trapdoor before dropping."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from probe_l9_boosted_supported_landing import report
from probe_l9_fast_high_switch_endgame import turned
from probe_l9_oneflip_trapdoors import yellows
from probe_l9_right_trap_stage import full_catches
from probe_l9_route_deletions import enter_level_9


def root_state(root):
    child = turned(root, 0)
    child.step(4)
    child.step(4)
    return child


def probe(env):
    enter_level_9(env)
    child = root_state(env)
    report(0, child)
    print("YELLOWS", yellows(child), "FULL", full_catches(child), flush=True)
    for index in range(1, 3):
        child.step(3)
        report(index, child)
        print("YELLOWS", yellows(child), "FULL", full_catches(child), flush=True)


if __name__ == "__main__":
    arena.run_program("bp35", probe)

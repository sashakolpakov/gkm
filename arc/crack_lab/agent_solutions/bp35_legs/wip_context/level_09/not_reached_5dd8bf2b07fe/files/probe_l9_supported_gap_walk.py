"""Trace horizontal movement from the depth-six column-six final landing."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from probe_l9_route_deletions import enter_level_9
from probe_l9_skip4_switch_choices import report
from probe_l9_supported_gap_climb import flipped


def probe(env):
    enter_level_9(env)
    for direction in (3, 4):
        child = flipped(env, 6)
        report((direction, 0), child)
        for index in range(1, 11):
            child.step(direction)
            report((direction, index), child)
            if child.terminal() or int(child.levels_completed) >= 9:
                break


if __name__ == "__main__":
    arena.run_program("bp35", probe)

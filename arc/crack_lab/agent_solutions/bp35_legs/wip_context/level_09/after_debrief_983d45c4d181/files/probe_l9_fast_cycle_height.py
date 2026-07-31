"""Reverse at an earlier compressed return-climb height."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from probe_l9_boosted_supported_landing import report
from probe_l9_control_row import controls
from probe_l9_fast_right_handoff import reversed_root
from probe_l9_route_deletions import enter_level_9


def probe(env):
    enter_level_9(env)
    height = int(sys.argv[1])
    switch_index = int(sys.argv[2])
    child = reversed_root(env)
    for _ in range(height):
        child.step(6, 27, 33)
    report((height, switch_index, "PRE"), child)
    visible = controls(child)
    child.step(*visible[switch_index])
    report((height, switch_index, "FLIP"), child)


if __name__ == "__main__":
    arena.run_program("bp35", probe)

"""Reverse one of the two controls at the compressed height-eight deadline."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from probe_l9_boosted_supported_landing import report
from probe_l9_control_row import controls
from probe_l9_faster_cycle import reversed_root
from probe_l9_route_deletions import enter_level_9


def height_eight(root):
    child = reversed_root(root)
    for _ in range(8):
        child.step(6, 27, 33)
    return child


def probe(env):
    enter_level_9(env)
    child = height_eight(env)
    switch = controls(child)[int(sys.argv[1])]
    child.step(*switch)
    report((sys.argv[1], switch, "FLIP"), child)


if __name__ == "__main__":
    arena.run_program("bp35", probe)

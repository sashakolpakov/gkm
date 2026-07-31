"""Find a catch interaction that safely exposes a fifth gravity control."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from probe_l9_boosted_supported_landing import report
from probe_l9_control_row import controls
from probe_l9_preserve_boost import boosted
from probe_l9_route_deletions import enter_level_9


def probe(env):
    enter_level_9(env)
    root = boosted(env)
    frame = root.frame()
    targets = (
        (x, y)
        for y in (3, 9, 15, 21, 27, 33, 39, 45, 51, 57)
        for x in (15, 21, 27, 33, 39)
        if int(frame[y][x]) in (12, 14, 15)
    )
    for target in targets:
        child = root.clone()
        child.step(6, *target)
        if not child.terminal():
            child.step(6, 27, 33)
        if not child.terminal():
            report((target, len(controls(child))), child)


if __name__ == "__main__":
    arena.run_program("bp35", probe)

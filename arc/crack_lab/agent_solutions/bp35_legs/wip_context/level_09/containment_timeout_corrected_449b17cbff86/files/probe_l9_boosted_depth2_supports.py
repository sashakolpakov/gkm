"""Extend the boosted catch tail immediately before its third descent."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from probe_l9_boosted_c4_descent import c4
from probe_l9_boosted_supported_landing import report
from probe_l9_route_deletions import enter_level_9


def depth2(root):
    child = c4(root)
    child.step(6, 27, 33)
    child.step(6, 27, 33)
    return child


def probe(env):
    enter_level_9(env)
    targets = tuple(
        (x, y)
        for y in (21, 27, 33, 39, 45, 51, 57)
        for x in (15, 21, 27, 33, 39)
    )
    for target in targets:
        child = depth2(env)
        child.step(6, *target)
        if not child.terminal():
            child.step(6, 27, 33)
        if not child.terminal():
            report((target, "SAFE"), child)


if __name__ == "__main__":
    arena.run_program("bp35", probe)

"""Check whether the four-control frontier can expose another lower switch."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from probe_l9_boosted_supported_landing import report
from probe_l9_preserve_boost import boosted
from probe_l9_route_deletions import enter_level_9


def probe(env):
    enter_level_9(env)
    child = boosted(env)
    report(0, child)
    for height in range(1, 9):
        child.step(6, 27, 33)
        report(height, child)
        if child.terminal() or int(child.levels_completed) >= 9:
            break


if __name__ == "__main__":
    arena.run_program("bp35", probe)

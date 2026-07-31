"""Check whether stack undo can restore a terminal local successor."""

import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from probe_l9_boosted_supported_landing import report
from probe_l9_preserve_boost import boosted
from probe_l9_route_deletions import enter_level_9


def key(env):
    frame = np.asarray(env.frame()).copy()
    frame[63, :] = 0
    return int(env.levels_completed), bool(env.terminal()), frame.tobytes()


def probe(env):
    enter_level_9(env)
    child = boosted(env)
    baseline = key(child)
    child.step(6, 27, 33)
    report("TERMINAL", child)
    child.step(7)
    print("RESTORED", key(child) == baseline, flush=True)
    report("UNDO", child)


if __name__ == "__main__":
    arena.run_program("bp35", probe)

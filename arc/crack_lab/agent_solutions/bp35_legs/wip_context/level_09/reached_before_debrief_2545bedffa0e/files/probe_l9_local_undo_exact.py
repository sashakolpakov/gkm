"""Verify action-seven restoration at the supported lower frontier."""

import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from probe_l9_boosted_supported_landing import report
from probe_l9_flipped_c5_supports import root_state
from probe_l9_route_deletions import enter_level_9


def key(env):
    frame = np.asarray(env.frame()).copy()
    frame[63, :] = 0
    return int(env.levels_completed), bool(env.terminal()), frame.tobytes()


def probe(env):
    enter_level_9(env)
    child = root_state(env)
    baseline = key(child)
    report("ROOT", child)
    for name, action in (
        ("CLICK", (6, 21, 27)),
        ("LEFT", 3),
        ("RIGHT", 4),
    ):
        child.step(*action) if isinstance(action, tuple) else child.step(action)
        report((name, "AFTER"), child)
        if child.terminal():
            print(name, "TERMINAL_NO_UNDO", flush=True)
            return
        child.step(7)
        print(name, "RESTORED", key(child) == baseline, flush=True)
        report((name, "UNDO"), child)


if __name__ == "__main__":
    arena.run_program("bp35", probe)

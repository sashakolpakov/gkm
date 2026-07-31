"""Verify exact stack restoration at the far-right lower-gap frontier."""

import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from probe_l9_route_deletions import enter_level_9
from probe_l9_supported_c9_actions import c9


def same(before, env):
    return bool(np.array_equal(before, np.asarray(env.frame())))


def probe(env):
    enter_level_9(env)
    root = c9(env)
    before = np.asarray(root.frame()).copy()
    root.step(6, 57, 45)
    print("SAFE_STEP", bool(root.terminal()), same(before, root), flush=True)
    root.step(7)
    print("SAFE_UNDO", bool(root.terminal()), same(before, root), flush=True)
    lethal = root.clone()
    lethal.step(6, 57, 33)
    print("LETHAL_STEP", bool(lethal.terminal()), same(before, lethal), flush=True)
    lethal.step(7)
    print("LETHAL_UNDO", bool(lethal.terminal()), same(before, lethal), flush=True)


if __name__ == "__main__":
    arena.run_program("bp35", probe)

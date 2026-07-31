"""Reach the former containment deadline 53 actions earlier and cross its endpoints."""

import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from probe_l9_early21_right import state
from probe_l9_early21_third_flips import third_flipped
from probe_l9_presecond_upper_height6_exits import height6 as old_height6
from probe_l9_route_deletions import enter_level_9


def early_height6(root):
    child = third_flipped(root, 4)
    child.step(6, 27, 33)
    for y in (3, 9, 15, 21, 27, 33):
        child.step(6, 45, y)
    child.step(6, 39, 33)
    child.step(6, 33, 33)
    child.step(6, 33, 39)
    child.step(4)
    for _ in range(6):
        child.step(6, 33, 33)
    return child


def physical(env):
    frame = np.asarray(env.frame()).copy()
    frame[63, :] = 0
    return frame


def apply(child, name, actions):
    for index, action in enumerate(actions, 1):
        child.step(*action) if isinstance(action, tuple) else child.step(action)
        state((name, index, action), child)
        if child.terminal() or int(child.levels_completed) >= 9:
            return


def probe(env):
    enter_level_9(env)
    root = early_height6(env)
    old = old_height6(env)
    state("EARLY_HEIGHT6", root)
    print(
        "COMPARE_OLD",
        int(np.count_nonzero(physical(root) != physical(old))),
        flush=True,
    )
    apply(
        root.clone(),
        "RIGHT",
        (
            (6, 39, 39), 4,
            (6, 45, 39), 4,
            (6, 51, 39), 4,
            (6, 57, 39), 4,
            4,
        ),
    )
    apply(
        root.clone(),
        "LEFT",
        (
            (6, 27, 39), 3,
            (6, 21, 39), 3,
            (6, 15, 39), 3,
            (6, 9, 39), 3,
            3,
        ),
    )


if __name__ == "__main__":
    arena.run_program("bp35", probe)

"""Test targeted extra deletions against the compressed boosted frontier."""

import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from probe_l9_control_row import compact, controls
from probe_l9_fast_boost_endgame import FAST_SKIPS
from probe_l9_route_deletions import enter_level_9, replay, route


TAIL = (
    (6, 21, 39),
    4,
    (6, 27, 39),
    4,
    (6, 27, 33),
    (6, 27, 33),
    (6, 27, 33),
)


def candidate(root, extra=()):
    child = replay(root, route(), skips=FAST_SKIPS | set(extra))
    if child.terminal():
        return child
    for action in TAIL:
        child.step(*action) if isinstance(action, tuple) else child.step(action)
        if child.terminal():
            break
    return child


def physical(env):
    frame = np.asarray(env.frame()).copy()
    frame[63, :] = 0
    return frame.tobytes()


def probe(env):
    enter_level_9(env)
    target = candidate(env)
    target_key = physical(target)
    variants = (
        (14,),
        (21,),
        (22,),
        (23,),
        (24,),
        (25,),
        (26,),
        (73,),
        (110,),
        (21, 22, 23, 24, 25, 26),
        (21, 22, 23, 24, 25, 26, 73, 110),
    )
    for extra in variants:
        child = candidate(env, extra)
        print(
            extra,
            "same",
            physical(child) == target_key,
            "terminal",
            bool(child.terminal()),
            "controls",
            controls(child),
            "grid",
            compact(child)["grid9"],
            flush=True,
        )


if __name__ == "__main__":
    arena.run_program("bp35", probe)

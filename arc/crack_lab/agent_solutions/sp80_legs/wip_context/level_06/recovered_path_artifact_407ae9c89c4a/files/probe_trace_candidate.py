"""Trace a single level-6 arrangement after each object move group."""
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as Arena

import perception as P
from players import (
    play_level_1, play_level_2, play_level_3, play_level_4, play_level_5,
)
from probe_l6_conditioned import POINTS


def movable(frame):
    return [
        (obj["color"], obj["bbox"], obj["area"])
        for obj in P.object_candidates(frame, min_area=8)
        if obj["color"] in (8, 9, 15)
    ]


def apply(env, actions):
    for action in actions:
        env.step(*action) if isinstance(action, tuple) else env.step(action)


def probe(env):
    for player in (
        play_level_1, play_level_2, play_level_3, play_level_4, play_level_5,
    ):
        player(env)
    print("TRACE", "START", movable(env.frame()))
    groups = (
        ("C_STAGE", [(6, *POINTS["C"])] + [3] * 3),
        ("D", [(6, *POINTS["D"])] + [4] + [1] * 6),
        ("A", [(6, *POINTS["A"])] + [2]),
        ("C", [(6, 15, 33)] + [4] * 3),
        ("B", [(6, *POINTS["B"])] + [2] * 4 + [3] * 2),
    )
    for name, actions in groups:
        apply(env, actions)
        print("TRACE", name, movable(env.frame()))
    env.step(5)
    print("TRACE", "USE", movable(env.frame()), "LEVEL", env.levels_completed)


if __name__ == "__main__":
    levels, path, err = Arena.run_program("sp80", probe)
    print("PROBE_RESULT", levels, len(path), err)

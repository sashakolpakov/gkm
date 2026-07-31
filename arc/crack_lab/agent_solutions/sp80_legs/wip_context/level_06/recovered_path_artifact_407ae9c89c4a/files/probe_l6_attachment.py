"""Check whether contact changes selection or movement grouping on level 6."""
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as Arena

import perception as P
from players import (
    play_level_1, play_level_2, play_level_3, play_level_4, play_level_5,
)


def movable(env):
    return tuple(
        (obj["color"], obj["bbox"], obj["area"])
        for obj in P.object_candidates(env.frame(), min_area=8)
        if obj["color"] in (8, 9, 15)
    )


def apply(env, actions):
    for action in actions:
        env.step(*action) if isinstance(action, tuple) else env.step(action)


def probe(env):
    for player in (
        play_level_1, play_level_2, play_level_3, play_level_4, play_level_5,
    ):
        player(env)
    cases = {
        "adjacent": [(6, 45, 18)] + [3] * 3,
        "overlap": [(6, 45, 18)] + [3] * 4,
    }
    for name, prefix in cases.items():
        node = env.clone()
        apply(node, prefix)
        staged = movable(node)
        apply(node, [(6, 30, 19), 2])
        print("ATTACH", name, "STAGED", staged, "AFTER_A_DOWN", movable(node))


if __name__ == "__main__":
    levels, path, err = Arena.run_program("sp80", probe)
    print("PROBE_RESULT", levels, len(path), err)

"""Check whether the USE indicator exposes partial arrangement progress."""
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

import perception as P
from players import (
    play_level_1, play_level_2, play_level_3, play_level_4, play_level_5,
)


def sample(env, path):
    node = env.clone()
    for action in path:
        node.step(*action) if isinstance(action, tuple) else node.step(action)
    before = P.arr(node.frame()).copy()
    node.step(5)
    after = P.arr(node.frame())
    delta = P.frame_delta(before, after)
    return (
        int(after[0, 63]), delta["count"], delta["bbox"],
        tuple(delta["samples"][-4:]), node.levels_completed,
    )


def probe(env):
    for player in (
        play_level_1, play_level_2, play_level_3, play_level_4, play_level_5,
    ):
        player(env)
    candidates = {
        "start": [],
        "central": (
            [(6, 30, 19), 2]
            + [(6, 33, 45)] + [1] * 3
            + [(6, 45, 18)] + [2] * 4 + [3] * 5
        ),
        "edge": (
            [(6, 30, 19)] + [2] * 4
            + [(6, 33, 45)] + [1] * 8 + [3]
            + [(6, 45, 18)] + [2] * 4 + [3] * 6
            + [(6, 25, 33)] + [3] * 3
        ),
    }
    print("USE_STATUS", {name: sample(env, path) for name, path in candidates.items()})


if __name__ == "__main__":
    levels, path, err = A.run_program("sp80", probe)
    print("PROBE_RESULT", levels, len(path), err)

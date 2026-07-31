import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A
import numpy as np

import perception as P
import players


def reach_level_five(env):
    for level in range(1, 5):
        getattr(players, f"play_level_{level}")(env)
        if env.levels_completed != level:
            print("advance_failed", level, env.levels_completed)
            return False
    return True


def summary(frame):
    blobs = P.connected_components(frame, min_area=4)
    return {
        "colors": P.color_counts(frame),
        "blobs": [(b.color, b.bbox, b.area) for b in blobs],
    }


def probe(env):
    if not reach_level_five(env):
        return
    base = np.asarray(env.frame()).copy()
    print("level", env.levels_completed + 1, "actions", env.actions)
    print("initial", summary(base))
    for action in (3, 4, 6, 7):
        clone = env.clone()
        try:
            clone.step(action)
            print(
                "key",
                action,
                "level",
                clone.levels_completed,
                "terminal",
                clone.terminal(),
                "delta",
                P.frame_delta(base, clone.frame()),
            )
        except Exception as exc:
            print("key", action, "error", type(exc).__name__, str(exc))


if __name__ == "__main__":
    A.run_program("bp35", probe)

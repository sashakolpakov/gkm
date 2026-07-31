import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A

import perception as P
import players


def summary(env):
    frame = env.frame()
    blobs = [
        (b.color, b.bbox, b.area)
        for b in P.connected_components(frame, min_area=3)
        if b.color != 1
    ]
    print("L6 actions", env.actions, "counts", P.color_counts(frame))
    print("blobs", blobs)
    for action in env.actions:
        clone = env.clone()
        before = clone.frame()
        level = clone.levels_completed
        clone.step(action)
        delta = P.frame_delta(before, clone.frame())
        print(
            "action", action,
            "delta", (delta["count"], delta["bbox"]),
            "reward", clone.levels_completed - level,
            "counts", P.color_counts(clone.frame()),
        )


def probe(env):
    for level in range(1, 6):
        getattr(players, f"play_level_{level}")(env)
    summary(env)


A.run_program("lf52", probe)

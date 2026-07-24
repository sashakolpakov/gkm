import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A
import numpy as np

from perception import color_counts, frame_delta, object_candidates


PATHS = (
    [2] * 10 + [3] * 5,
    [3] * 2 + [5] + [2] * 8,
    [5] + [2] * 7 + [4] * 7
    + [5] + [2] * 5 + [3] * 12
    + [5] + [1] * 16,
    [5] + [4] * 7
    + [5] + [4] * 7
    + [5] + [2] * 9,
)


def transitions(before, after):
    changed = before != after
    pairs, counts = np.unique(
        np.stack((before[changed], after[changed]), axis=1),
        axis=0,
        return_counts=True,
    )
    return {f"{int(a)}>{int(b)}": int(n) for (a, b), n in zip(pairs, counts)}


def key_objects(frame):
    return [
        (obj["color"], obj["bbox"], obj["area"])
        for obj in object_candidates(frame)
        if obj["color"] not in (4, 9)
    ]


def probe(env):
    for level, path in enumerate(PATHS, 1):
        for index, action in enumerate(path):
            before = np.asarray(env.frame()).copy()
            before_reward = env.levels_completed
            env.step(action)
            if env.levels_completed > before_reward:
                after = np.asarray(env.frame()).copy()
                print("WIN", level, "index", index, "action", action)
                print("BEFORE_COLORS", color_counts(before))
                print("BEFORE_OBJECTS", key_objects(before))
                print("DELTA", frame_delta(before, after))
                print("TRANSITIONS", transitions(before, after))


levels, path, err = A.run_program("ar25", probe)
print("PROBE_RESULT", levels, len(path), err)

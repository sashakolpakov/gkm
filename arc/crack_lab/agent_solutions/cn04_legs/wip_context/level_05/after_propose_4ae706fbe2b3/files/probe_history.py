import numpy as np

import gkm_try as harness
from perception import color_counts, connected_components


PATHS = [
    [2] * 7 + [4] * 4 + [5] * 3,
    [1] * 2 + [4] * 4 + [[6, 23, 35]] + [1] * 8 + [4] * 4
    + [[6, 50, 53]] + [5] * 3 + [1] * 10,
]


def summary(env):
    frame = np.asarray(env.frame())
    bg = max(color_counts(frame), key=color_counts(frame).get)
    blobs = [(b.color, b.bbox, b.area) for b in connected_components(frame, min_area=9)
             if b.color != bg and b.bbox[0] != 0]
    return int(np.count_nonzero(frame[1:] != bg)), blobs


def probe(env):
    for level, path in enumerate(PATHS, 1):
        print("L", level, "START", summary(env))
        for i, action in enumerate(path):
            before = summary(env)
            env.step(action)
            after = summary(env)
            if action == 5 or isinstance(action, list) or env.levels_completed == level:
                print(" step", i + 1, action, before[0], "->", after[0],
                      "reward", env.levels_completed)
            if env.levels_completed == level:
                break


harness.A.run_program("cn04", probe)

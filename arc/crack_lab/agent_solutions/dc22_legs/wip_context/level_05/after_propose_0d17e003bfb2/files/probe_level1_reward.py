import os
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A
import gkm_legs as G

from perception import arr


PATH = (
    (6, 48, 36),
    1, 1, 1, 1,
    4, 4, 4, 4, 4,
    (6, 48, 19),
    (6, 48, 36),
    1, 1, 1, 1, 1, 1,
    4, 4,
)


def centers(frame, colors=(11, 13, 14)):
    f = arr(frame)
    out = {}
    for color in colors:
        ys, xs = np.where(f[:56] == color)
        out[color] = None if not len(ys) else (
            round(float(ys.mean()), 1), round(float(xs.mean()), 1),
            int(len(ys)),
        )
    return out


def probe(env):
    print("start", env.levels_completed, centers(env.frame()))
    for index, action in enumerate(PATH, 1):
        before = env.levels_completed
        if isinstance(action, tuple):
            env.step(*action)
        else:
            env.step(action)
        if env.levels_completed != before or index >= len(PATH) - 2:
            print(index, action, env.levels_completed, centers(env.frame()))


if G._workspace_taint_reason(os.getcwd()):
    raise SystemExit("tainted")
A.run_program("dc22", probe)

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena
import numpy as np

import players


LEVEL_1 = [2] * 7 + [4] * 4 + [5] * 3
LEVEL_5 = (
    [5] * 4 + [1] * 3 + [4] * 6 + [(6, 54, 6)] + [5] * 3
    + [(6, 5, 38)] + [1] * 10 + [4] * 10
    + [(6, 47, 47)] + [5] * 3 + [1] * 10 + [4] * 2
)


def step(env, action):
    env.step(*action) if isinstance(action, tuple) else env.step(action)


def logical(frame, colored=False):
    a = np.asarray(frame)
    bg = int(np.bincount(a.ravel()).argmax())
    rows = []
    for r in range(3, 64, 3):
        line = ""
        for c in range(0, 64, 3):
            v = int(a[r, c])
            if v == bg:
                line += "."
            elif colored and v not in (0, 4):
                line += "S"
            else:
                line += "#"
        rows.append(line)
    used = [(i, s) for i, s in enumerate(rows, 1) if "#" in s]
    if not used:
        return
    lo = min(s.index("#") for _, s in used)
    hi = max(s.rindex("#") for _, s in used)
    for r, line in used:
        print(f"{r:02} {line[lo:hi + 1]}")


def probe(env):
    print("L1_INITIAL")
    logical(env.frame())
    for action in LEVEL_1[:-1]:
        step(env, action)
    print("L1_PREWIN")
    logical(env.frame())
    step(env, LEVEL_1[-1])

    while env.levels_completed < 4:
        getattr(players, f"play_level_{env.levels_completed + 1}")(env)
    print("L5_INITIAL")
    logical(env.frame())
    for action in LEVEL_5[:-1]:
        step(env, action)
    print("L5_PREWIN")
    logical(env.frame(), colored=True)


arena.run_program("cn04", probe)

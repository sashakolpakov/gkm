import importlib.util
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A


spec = importlib.util.spec_from_file_location("solve", "solve.py")
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)


R = (6, 56, 28)
S = (6, 46, 28)
PATH = [
    2, 4,
    R, 4, R, 4, R, 4, R,
    4, 4, 4, 4, 4, 4,
    2, 2, 2, 2, 4,
    R, R, R, R,
    S, S, S, S, S,
    3, 3, 3, 3, 2,
    S, S,
    1, 3, 3, 3, 3, 3,
    2, 2, 2, 2,
    S, S, S, S, S,
    2,
    S, S, S, S, S,
    2,
    4, 4, 4, 4, 4, 4,
    2, 2,
]

FERRY_PATH = [
    2, 4,
    R, 4, R, 4, R, 4, R,
    4, 4, 4, 4, 4, 4,
    2, 2, 2, 2, 4,
    R, R, R, R,
    S, S, S, S, S,
    3, 3, 3, 3, 2, 2,
    (6, 52, 19),
    S,
    2, 4, 4, 2, 2, 2, 2, 2, 2,
    S, 4, S, 4, S, 4, S, 4,
    2, 4, 4, 4, 4, 4, 4, 2, 2, 4,
]


def avatar(env):
    ys, xs = np.where(np.asarray(env.frame()) == 14)
    return None if not len(ys) else (int(ys.min()), int(xs.min()))


def probe(env):
    solver.solve(env)
    base = int(env.levels_completed)
    print("START", base, avatar(env))
    for i, action in enumerate(FERRY_PATH, 1):
        env.step(*action) if isinstance(action, tuple) else env.step(action)
        if i >= 20:
            print("STEP", i, action, avatar(env), int(env.levels_completed))
        if env.levels_completed > base:
            print("FOUND", i, FERRY_PATH[:i], avatar(env))
            return
    print("FAIL", avatar(env), int(env.levels_completed))


levels, path, err = A.run_program("dc22", probe)
print("END", levels, len(path), err)

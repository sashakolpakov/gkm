import importlib.util
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A


spec = importlib.util.spec_from_file_location("solve", "solve.py")
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)

C = (6, 50, 28)
D = (6, 56, 28)
DOCK = (6, 52, 35)
LEFT = (6, 44, 29)
RIGHT = (6, 54, 29)
DOWN = (6, 60, 29)
UP = (6, 50, 29)
E = (6, 52, 42)
F = (6, 52, 46)


def avatar(env):
    ys, xs = np.where(np.asarray(env.frame()) == 14)
    return None if not len(ys) else (int(ys.min()), int(xs.min()))


def do(env, actions, label):
    for action in actions:
        env.step(*action) if isinstance(action, tuple) else env.step(action)
    print(label, avatar(env), int(env.levels_completed))


def probe(env):
    solver.solve(env)
    do(env, [C] * 3 + [D] * 3 + [DOCK], "DOCK")
    do(env, [LEFT] * 3 + [DOWN] * 3 + [RIGHT] * 3, "LOWER_RIGHT")
    do(env, [3, E, 3, 1, 1, 3, 3] + [1] * 4 + [4, 1],
       "WAITING")
    do(env, [LEFT] * 3 + [UP] * 3 + [RIGHT] * 3, "TOP_RIGHT")
    do(env, [3] + [1] * 7, "TOP_PORTAL")
    do(env, [F], "TELEPORTED")
    do(env, [E], "LOWER_BRIDGE")
    do(env, [2] * 5, "LOWER_END")


levels, path, err = A.run_program("dc22", probe)
print("END", levels, len(path), err)

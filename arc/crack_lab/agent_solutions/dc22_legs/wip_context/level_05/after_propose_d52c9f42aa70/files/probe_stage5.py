import importlib.util
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A


spec = importlib.util.spec_from_file_location("solve", "solve.py")
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)

E = (6, 52, 42)
F = (6, 52, 46)


def avatar(env):
    ys, xs = np.where(np.asarray(env.frame()) == 14)
    return None if not len(ys) else (int(ys.min()), int(xs.min()))


def run(env, label, path):
    clone = env.clone()
    out = [avatar(clone)]
    for action in path:
        clone.step(*action) if isinstance(action, tuple) else clone.step(action)
        out.append(avatar(clone))
    print(label, "POS", out, "LEVEL", int(clone.levels_completed))
    return clone


def probe(env):
    solver.solve(env)
    print("BASE", avatar(env), int(env.levels_completed))
    run(env, "UPPER_BRIDGE", [3, E, 3, E])
    west = run(env, "LOWER_BRIDGE", [2, 3, E, 3, 3, E])
    run(west, "TO_6_F", [1, 1, F])
    run(west, "TO_6_F_STEP", [1, 1, F, 1, 2])


levels, path, err = A.run_program("dc22", probe)
print("END", levels, len(path), err)

import importlib.util
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A


spec = importlib.util.spec_from_file_location("solve", "solve.py")
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)


def avatar(frame):
    ys, xs = np.where(np.asarray(frame) == 14)
    return None if not len(ys) else (int(ys.min()), int(xs.min()))


def probe(env):
    solver.solve(env)
    before = np.asarray(env.frame()).copy()
    for index in range(1, 61):
        action = 2 if index % 2 else 1
        env.step(action)
        after = np.asarray(env.frame()).copy()
        mask = before != after
        # Ignore avatar's old/new 4x4 neighborhood and status row.
        mask[30:40, 28:36] = False
        mask[63, :] = False
        if mask.any():
            ys, xs = np.where(mask)
            print("WORLD", index, action, int(mask.sum()),
                  (int(ys.min()), int(xs.min()),
                   int(ys.max()), int(xs.max())))
        before = after
    print("FINAL", avatar(env.frame()), int(env.levels_completed))


levels, path, err = A.run_program("dc22", probe)
print("END", levels, len(path), err)

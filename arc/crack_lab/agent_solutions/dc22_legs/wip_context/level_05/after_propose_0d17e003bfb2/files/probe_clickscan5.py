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
    base = env.frame()
    hits = []
    for y in range(1, 64, 2):
        for x in range(1, 64, 2):
            clone = env.clone()
            clone.step(6, x, y)
            before = np.asarray(base)
            after = np.asarray(clone.frame())
            mask = before != after
            mask[63, :] = False
            if mask.any():
                ys, xs = np.where(mask)
                hits.append((x, y, int(before[y, x]), int(mask.sum()),
                             (int(ys.min()), int(xs.min()),
                              int(ys.max()), int(xs.max())),
                             avatar(clone.frame())))
    print("HITS", hits)


levels, path, err = A.run_program("dc22", probe)
print("END", levels, len(path), err)

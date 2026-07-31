import importlib.util
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

from perception import frame_delta


spec = importlib.util.spec_from_file_location("solve", "solve.py")
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)


def bbox(frame, color):
    a = np.asarray(frame)
    ys, xs = np.where(a[:56, :38] == color)
    return None if not len(ys) else (
        int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max()))


def probe(env):
    solver.solve(env)
    base = env.frame()
    for y in (28, 29, 30, 31):
        for x in range(42, 63, 2):
            clone = env.clone()
            clone.step(6, x, y)
            after = clone.frame()
            delta = frame_delta(base, after)
            if delta["count"] > 1:
                print("HIT", x, y, int(np.asarray(base)[y, x]),
                      delta["count"], delta["bbox"], bbox(after, 8))


levels, path, err = A.run_program("dc22", probe)
print("END", levels, len(path), err)

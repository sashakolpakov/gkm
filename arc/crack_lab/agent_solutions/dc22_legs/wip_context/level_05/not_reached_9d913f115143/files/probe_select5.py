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
    a = np.asarray(frame)[4:56, :38]
    ys, xs = np.where(a == color)
    return None if not len(ys) else (
        int(ys.min() + 4), int(xs.min()),
        int(ys.max() + 4), int(xs.max()))


def point(frame, color):
    a = np.asarray(frame)[4:56, :38]
    ys, xs = np.where(a == color)
    return int(xs[0]), int(ys[0] + 4)


def probe(env):
    solver.solve(env)
    base = env.frame()
    for color in (14, 8, 6, 9, 11, 15, 2, 4):
        click = point(base, color)
        for action in (1, 2, 3, 4):
            clone = env.clone()
            clone.step(6, *click)
            clicked = clone.frame()
            clone.step(action)
            after = clone.frame()
            delta = frame_delta(clicked, after)
            print("TEST", color, click, action,
                  "CLICK", frame_delta(base, clicked)["count"],
                  "D", delta["count"], delta["bbox"],
                  "AV", bbox(after, 14), "P", bbox(after, 8))


levels, path, err = A.run_program("dc22", probe)
print("END", levels, len(path), err)

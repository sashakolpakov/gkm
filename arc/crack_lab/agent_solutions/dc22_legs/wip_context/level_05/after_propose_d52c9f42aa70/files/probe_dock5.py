import importlib.util
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

from perception import frame_delta


spec = importlib.util.spec_from_file_location("solve", "solve.py")
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)

C = (6, 50, 28)
D = (6, 56, 28)
G = (6, 52, 35)


def platform(frame):
    a = np.asarray(frame)[:56, :38]
    ys, xs = np.where(a == 8)
    return (
        int(len(ys)),
        None if not len(ys) else (
            int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())))


def probe(env):
    solver.solve(env)
    for action in (C, C, C, D, D, D):
        env.step(*action)
    before = env.frame()
    print("DOCK", platform(before), int(env.levels_completed))
    for phase in range(1, 9):
        env.step(*G)
        after = env.frame()
        d = frame_delta(before, after)
        print("GLYPH", phase, platform(after), d["count"], d["bbox"],
              int(env.levels_completed))
        before = after


levels, path, err = A.run_program("dc22", probe)
print("END", levels, len(path), err)

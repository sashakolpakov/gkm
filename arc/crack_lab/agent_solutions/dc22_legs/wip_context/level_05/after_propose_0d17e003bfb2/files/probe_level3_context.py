import importlib.util
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

from perception import frame_delta


spec = importlib.util.spec_from_file_location("solve", "solve.py")
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)

C1 = (6, 51, 18)
C2 = (6, 51, 27)
C3 = (6, 51, 36)
C4 = (6, 51, 45)
TO_A = [3, C1, C2, 3, 3, 3, 2, 2, 2, 3, C1, 3, 3, 3, 3,
        C2, 3, 3, 3, C2, 1, 1, 1]
TO_B = TO_A + [1, C3]
TO_15 = TO_B + [1, 1, 4, 4]


def do(env, action):
    env.step(*action) if isinstance(action, tuple) else env.step(action)


def avatar(frame):
    ys, xs = np.where(np.asarray(frame) == 14)
    return None if not len(ys) else (int(ys.min()), int(xs.min()))


def summary(env, label, reference=None):
    f = np.asarray(env.frame())
    values = {
        color: int(np.count_nonzero(f == color))
        for color in (6, 7, 11, 14, 15)
    }
    delta = None if reference is None else frame_delta(reference, f)
    print(label, "LEVEL", env.levels_completed, "AV", avatar(f),
          "COUNTS", values,
          "DELTA", None if delta is None else (delta["count"], delta["bbox"]))


def probe(env):
    solver.solve(env)
    root = env.clone()
    initial = np.asarray(root.frame()).copy()
    summary(root, "ROOT")
    for label, path in (("A", TO_A), ("B", TO_B), ("ON15", TO_15)):
        node = root.clone()
        for action in path:
            do(node, action)
        summary(node, label, initial)
        if label == "ON15":
            for action in (1, 2, 3, 4, C1, C2, C3, C4):
                child = node.clone()
                do(child, action)
                summary(child, f"ON15+{action}", np.asarray(node.frame()).copy())


levels, path, err = A.run_program("dc22", probe)
print("END", levels, len(path), err)

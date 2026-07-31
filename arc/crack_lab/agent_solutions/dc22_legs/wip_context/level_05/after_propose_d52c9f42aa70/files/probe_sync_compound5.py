import importlib.util
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

from perception import connected_components, frame_delta


spec = importlib.util.spec_from_file_location("solve", "solve.py")
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)

C = (6, 50, 28)
D = (6, 56, 28)
DOCK = (6, 52, 35)
X = (6, 44, 29)
Y = (6, 54, 29)
Z = (6, 60, 29)
E = (6, 52, 42)
STAGE = (C, C, C, D, D, D, DOCK, X, X, X,
         Z, Z, Z, Y, Y, Y, 3, E, 3, 1, 1, 3)


def avatar(frame):
    ys, xs = np.where(np.asarray(frame) == 14)
    return None if not len(ys) else (int(ys.min()), int(xs.min()))


def compound(frame):
    return [
        (b.color, b.bbox, b.area)
        for b in connected_components(
            np.asarray(frame)[:56, :38], colors=(8, 12))
    ]


def run(env, label, approach):
    node = env.clone()
    for action in STAGE + approach:
        node.step(*action) if isinstance(action, tuple) else node.step(action)
    before = node.frame()
    print("ROOT", label, avatar(before), compound(before))
    for phase in range(1, 6):
        node.step(*X)
        middle = node.frame()
        dc = frame_delta(before, middle)
        node.step(3)
        after = node.frame()
        ds = frame_delta(middle, after)
        print("PHASE", label, phase, avatar(after), compound(after),
              "CLICK", dc["count"], dc["bbox"],
              "STEP", ds["count"], ds["bbox"])
        before = after


def probe(env):
    solver.solve(env)
    run(env, "LOWER", (3,))
    run(env, "UPPER", (3, 1, 1, 1, 1))


levels, path, err = A.run_program("dc22", probe)
print("END", levels, len(path), err)

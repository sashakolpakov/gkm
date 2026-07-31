import importlib.util
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

from perception import connected_components, frame_delta


spec = importlib.util.spec_from_file_location("solve", "solve.py")
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)

E = (6, 52, 42)


def avatar(frame):
    ys, xs = np.where(np.asarray(frame) == 14)
    return None if not len(ys) else (int(ys.min()), int(xs.min()))


def bridges(frame):
    return [(b.bbox, b.area) for b in connected_components(
        np.asarray(frame)[:56, :38], colors=(9,))]


def probe(env):
    solver.solve(env)
    before = env.frame()
    print("S", avatar(before), bridges(before))
    for i, action in enumerate([3, E, 3, E, 3, E, 3, E], 1):
        env.step(*action) if isinstance(action, tuple) else env.step(action)
        after = env.frame()
        d = frame_delta(before, after)
        print(i, "E" if isinstance(action, tuple) else action,
              avatar(after), bridges(after), d["count"], d["bbox"])
        before = after


levels, path, err = A.run_program("dc22", probe)
print("END", levels, len(path), err)

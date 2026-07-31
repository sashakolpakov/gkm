import importlib.util
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

from perception import frame_delta


spec = importlib.util.spec_from_file_location("solve", "solve.py")
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)


def avatar(frame):
    ys, xs = np.where(np.asarray(frame) == 14)
    return None if not len(ys) else (int(ys.min()), int(xs.min()))


PAIRS = {
    "avatar_left": ((33, 35), (31, 35)),
    "avatar_goal": ((33, 35), (11, 51)),
    "platform_target": ((5, 25), (11, 19)),
    "bridge_upper": ((11, 39), (27, 35)),
    "endpoint_pair": ((11, 35), (25, 7)),
}


def probe(env):
    solver.solve(env)
    base = env.frame()
    for label, (first, second) in PAIRS.items():
        clone = env.clone()
        clone.step(6, *first)
        middle = clone.frame()
        clone.step(6, *second)
        after = clone.frame()
        print(label, frame_delta(base, middle),
              frame_delta(middle, after), avatar(after),
              int(clone.levels_completed))


levels, path, err = A.run_program("dc22", probe)
print("END", levels, len(path), err)

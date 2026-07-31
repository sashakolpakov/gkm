import importlib.util
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A


spec = importlib.util.spec_from_file_location("solve", "solve.py")
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)

E = (6, 52, 42)
CONTEXTS = {
    "BASE": (),
    "ADJ": (3, E),
    "ON": (3, E, 3, 3),
    "TOP_EAST": (3, E, 3, 1, 1),
    "TOP_WEST": (3, E, 3, 3, 1, 1),
}


def avatar(frame):
    ys, xs = np.where(np.asarray(frame) == 14)
    return None if not len(ys) else (int(ys.min()), int(xs.min()))


def probe(env):
    solver.solve(env)
    for label, path in CONTEXTS.items():
        root = env.clone()
        for action in path:
            root.step(*action) if isinstance(action, tuple) else root.step(action)
        expected = {}
        for action in (1, 2, 3, 4):
            direct = root.clone()
            direct.step(action)
            expected[action] = np.asarray(direct.frame())[:56].copy()
        hits = []
        for y in range(1, 56, 2):
            for x in range(1, 38, 2):
                for action in (1, 2, 3, 4):
                    clone = root.clone()
                    clone.step(6, x, y)
                    clone.step(action)
                    after = np.asarray(clone.frame())[:56]
                    if not np.array_equal(after, expected[action]):
                        hits.append((
                            x, y, int(np.asarray(root.frame())[y, x]),
                            action, avatar(after)))
        print("CONTEXT", label, avatar(root.frame()), "HITS", hits)


levels, path, err = A.run_program("dc22", probe)
print("END", levels, len(path), err)

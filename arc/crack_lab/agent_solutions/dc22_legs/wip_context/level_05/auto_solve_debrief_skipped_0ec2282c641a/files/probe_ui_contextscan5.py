import importlib.util
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A


spec = importlib.util.spec_from_file_location("solve", "solve.py")
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)

A_CTL = (6, 46, 22)
B_CTL = (6, 56, 22)
C = (6, 50, 28)
D = (6, 56, 28)
E = (6, 52, 42)
F = (6, 52, 46)
CONTEXS_X = (6, 44, 29)
CONTEXTS = {
    "LOWER_LEFT": (
        C, C, C, D, D, D, (6, 52, 35),
        CONTEXS_X, CONTEXS_X, CONTEXS_X,
        (6, 60, 29), (6, 60, 29), (6, 60, 29)),
}


def avatar(frame):
    ys, xs = np.where(np.asarray(frame) == 14)
    return None if not len(ys) else (int(ys.min()), int(xs.min()))


def probe(env):
    solver.solve(env)
    for label, path in CONTEXTS.items():
        root = env.clone()
        for action in path:
            root.step(*action)
        before = np.asarray(root.frame())[:56].copy()
        direct = {}
        for action in (1, 2, 3, 4):
            child = root.clone()
            child.step(action)
            direct[action] = np.asarray(child.frame())[:56].copy()
        visible = []
        hidden = []
        for y in range(1, 56, 2):
            for x in range(39, 64, 2):
                clicked = root.clone()
                clicked.step(6, x, y)
                click_frame = np.asarray(clicked.frame())[:56]
                if not np.array_equal(click_frame, before):
                    mask = click_frame != before
                    ys, xs = np.where(mask)
                    visible.append((
                        x, y, int(np.asarray(root.frame())[y, x]),
                        int(mask.sum()),
                        (int(ys.min()), int(xs.min()),
                         int(ys.max()), int(xs.max()))))
                    continue
                for action in (1, 2, 3, 4):
                    child = clicked.clone()
                    child.step(action)
                    after = np.asarray(child.frame())[:56]
                    if not np.array_equal(after, direct[action]):
                        hidden.append((x, y, action, avatar(after)))
        print("CONTEXT", label, "VISIBLE", visible, "HIDDEN", hidden)


levels, path, err = A.run_program("dc22", probe)
print("END", levels, len(path), err)

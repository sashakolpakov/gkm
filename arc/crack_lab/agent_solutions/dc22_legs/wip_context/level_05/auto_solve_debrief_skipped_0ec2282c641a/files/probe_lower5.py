import importlib.util
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A


spec = importlib.util.spec_from_file_location("solve", "solve.py")
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)

C = (6, 50, 28)
D = (6, 56, 28)
DOCK = (6, 52, 35)
LEFT = (6, 44, 29)
RIGHT = (6, 54, 29)
DOWN = (6, 60, 29)
UP = (6, 50, 29)
E = (6, 52, 42)
F = (6, 52, 46)
A_CTL = (6, 46, 22)
B_CTL = (6, 56, 22)


def avatar(frame):
    ys, xs = np.where(np.asarray(frame) == 14)
    return None if not len(ys) else (int(ys.min()), int(xs.min()))


def do(env, action):
    env.step(*action) if isinstance(action, tuple) else env.step(action)


def stage(env):
    path = (
        [C] * 3 + [D] * 3 + [DOCK] +
        [LEFT] * 3 + [DOWN] * 3 + [RIGHT] * 3 +
        [3, E, 3, 1, 1, 3, 3] + [1] * 4 + [4, 1] +
        [LEFT] * 3 + [UP] * 3 + [RIGHT] * 3 +
        [3] + [1] * 7 + [F, E] + [2] * 5 + [4])
    for action in path:
        do(env, action)


def probe(env):
    solver.solve(env)
    stage(env)
    print("START", avatar(env.frame()))
    for phase in range(1, 5):
        do(env, B_CTL)
        do(env, 4)
        print("B", phase, avatar(env.frame()))
    for _ in range(4):
        do(env, 4)
        print("RIGHT", avatar(env.frame()))
    root = env.clone()
    before = np.asarray(root.frame())[:56].copy()
    hits = []
    for y in range(1, 56, 2):
        for x in range(39, 64, 2):
            child = root.clone()
            child.step(6, x, y)
            after = np.asarray(child.frame())[:56]
            if not np.array_equal(after, before):
                mask = after != before
                ys, xs = np.where(mask)
                hits.append((
                    x, y, int(np.asarray(root.frame())[y, x]),
                    int(mask.sum()),
                    (int(ys.min()), int(xs.min()),
                     int(ys.max()), int(xs.max()))))
    print("ENDPOINT", avatar(root.frame()), "HITS", hits)
    for action in (1, 2, 3, 4, A_CTL, B_CTL, E, F):
        child = root.clone()
        do(child, action)
        print("ACT", action, avatar(child.frame()),
              int(child.levels_completed))
    node = root.clone()
    for phase in range(1, 7):
        do(node, A_CTL)
        do(node, 2)
        print("A_DOWN", phase, avatar(node.frame()),
              int(node.levels_completed))
    docked = root.clone()
    for _ in range(5):
        do(docked, A_CTL)
    before = np.asarray(docked.frame())[:56].copy()
    dock_hits = []
    for y in range(1, 56, 2):
        for x in range(39, 64, 2):
            child = docked.clone()
            child.step(6, x, y)
            after = np.asarray(child.frame())[:56]
            if not np.array_equal(after, before):
                mask = after != before
                ys, xs = np.where(mask)
                dock_hits.append((
                    x, y, int(np.asarray(docked.frame())[y, x]),
                    int(mask.sum()),
                    (int(ys.min()), int(xs.min()),
                     int(ys.max()), int(xs.max()))))
    print("A_DOCK_HITS", dock_hits)
    for label, scan_root in (("ENDPOINT", root), ("A_DOCK", docked)):
        before = np.asarray(scan_root.frame())[:56].copy()
        game_hits = []
        for y in range(1, 56, 2):
            for x in range(1, 38, 2):
                child = scan_root.clone()
                child.step(6, x, y)
                after = np.asarray(child.frame())[:56]
                if not np.array_equal(after, before):
                    mask = after != before
                    ys, xs = np.where(mask)
                    game_hits.append((
                        x, y, int(np.asarray(scan_root.frame())[y, x]),
                        int(mask.sum()),
                        (int(ys.min()), int(xs.min()),
                         int(ys.max()), int(xs.max()))))
        print("GAME_HITS", label, game_hits)
    vacated = root.clone()
    for _ in range(9):
        do(vacated, 3)
    for _ in range(5):
        do(vacated, A_CTL)
    before = np.asarray(vacated.frame())[:56].copy()
    vacated_hits = []
    for y in range(1, 56, 2):
        for x in range(39, 64, 2):
            child = vacated.clone()
            child.step(6, x, y)
            after = np.asarray(child.frame())[:56]
            if not np.array_equal(after, before):
                mask = after != before
                ys, xs = np.where(mask)
                vacated_hits.append((
                    x, y, int(np.asarray(vacated.frame())[y, x]),
                    int(mask.sum()),
                    (int(ys.min()), int(xs.min()),
                     int(ys.max()), int(xs.max()))))
    print("VACATED", avatar(vacated.frame()), "HITS", vacated_hits)


levels, path, err = A.run_program("dc22", probe)
print("END", levels, len(path), err)

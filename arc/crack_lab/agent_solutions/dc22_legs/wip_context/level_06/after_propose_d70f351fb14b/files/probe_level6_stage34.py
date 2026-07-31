import importlib.util
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A
import gkm_legs as G

from perception import frame_delta


taint_reason = G._workspace_taint_reason(os.getcwd())
if taint_reason:
    raise SystemExit(f"TAINTED WORKSPACE: {taint_reason}")

spec = importlib.util.spec_from_file_location("solve", "solve.py")
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)

A_CONTROL = (6, 56, 8)
B_CONTROL = (6, 51, 25)
REMOTE = [
    3, 3, 3, 3, 3,
    A_CONTROL, A_CONTROL, A_CONTROL, A_CONTROL,
    2, 2, 3, 3, 3, 2, 3, A_CONTROL, 1, A_CONTROL, 1, 1,
    B_CONTROL,
] + [1] * 17 + [3] + [2] * 11
STAGED = (
    REMOTE
    + [2] * 8
    + [4, 4, A_CONTROL, 4, A_CONTROL, 1]
    + [A_CONTROL, 4] * 3
    + [1, 1, 1]
    + [
        (6, 51, 48), (6, 51, 48), (6, 51, 48), 3, B_CONTROL,
        4, 4, (6, 54, 36), 3, B_CONTROL, 4,
    ]
)


def apply(env, path):
    for action in path:
        env.step(*action) if isinstance(action, tuple) else env.step(action)


def avatar_tile(env):
    frame = env.frame()
    for row in range(31):
        for col in range(32):
            if (frame[2 * row:2 * row + 2, 2 * col:2 * col + 2] == 14).all():
                return row, col
    return None


def run(env):
    solver.solve(env)
    apply(env, STAGED)
    base = env.frame().copy()
    groups = {}
    for y in range(0, 63, 2):
        for x in range(0, 64, 2):
            child = env.clone()
            child.step(6, x, y)
            delta = frame_delta(base[:63], child.frame()[:63])
            if delta["count"] == 0:
                continue
            key = (delta["count"], delta["bbox"], avatar_tile(child), child.levels_completed)
            groups.setdefault(key, []).append((x, y))
    for effect, points in groups.items():
        print("EFFECT", effect, "points", points)


A.run_program("dc22", run)

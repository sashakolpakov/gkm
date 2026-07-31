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
S_CONTROL = (6, 51, 48)
D_LEFT = (6, 46, 36)
REMOTE = [
    3, 3, 3, 3, 3,
    A_CONTROL, A_CONTROL, A_CONTROL, A_CONTROL,
    2, 2, 3, 3, 3, 2, 3, A_CONTROL, 1, A_CONTROL, 1, 1,
    B_CONTROL,
] + [1] * 17 + [3] + [2] * 11
SELECTOR = (
    REMOTE
    + [2] * 8
    + [4, 4, A_CONTROL, 4, A_CONTROL, 1]
    + [A_CONTROL, 4] * 3
    + [1, 1, 1, S_CONTROL, S_CONTROL, 3, B_CONTROL]
)


def apply(env, path):
    for action in path:
        env.step(*action) if isinstance(action, tuple) else env.step(action)


def tile(env, row, col):
    frame = env.frame()
    return tuple(int(v) for v in frame[
        2 * row:2 * row + 2, 2 * col:2 * col + 2
    ].ravel())


def endpoints(env):
    return {
        "selector": tile(env, 24, 9),
        "top": tile(env, 2, 2),
        "lower": tile(env, 26, 16),
        "hub": tile(env, 29, 17),
        "hub_left": tile(env, 29, 16),
    }


def run(env):
    solver.solve(env)
    apply(env, SELECTOR)
    print("TOP_ARRIVAL", endpoints(env))
    env.step(*S_CONTROL)
    print("SELECT_3", endpoints(env))
    before = env.frame().copy()
    env.step(*B_CONTROL)
    delta = frame_delta(before[:63], env.frame()[:63])
    print(
        "MISMATCH_B", endpoints(env),
        "delta", (delta["count"], delta["bbox"]),
        "level", env.levels_completed,
    )
    for direction in (1, 2, 3, 4):
        child = env.clone()
        before = child.frame().copy()
        child.step(direction)
        delta = frame_delta(before[:63], child.frame()[:63])
        print("MOVE", direction, (delta["count"], delta["bbox"]), endpoints(child))
    before = env.frame().copy()
    env.step(*D_LEFT)
    delta = frame_delta(before[:63], env.frame()[:63])
    print("D_LEFT", (delta["count"], delta["bbox"]), endpoints(env))


A.run_program("dc22", run)

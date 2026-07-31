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
    + [1, 1, 1, S_CONTROL, 3]
)


def apply(env, path):
    for action in path:
        env.step(*action) if isinstance(action, tuple) else env.step(action)


def run(env):
    solver.solve(env)
    apply(env, SELECTOR)
    for activated in (False, True):
        root = env.clone()
        if activated:
            root.step(*B_CONTROL)
        print("ACTIVATED", activated)
        for direction in (1, 2, 3, 4):
            child = root.clone()
            before = child.frame().copy()
            child.step(direction)
            delta = frame_delta(before[:63], child.frame()[:63])
            print(
                "DIR", direction,
                "delta", (delta["count"], delta["bbox"]),
                "samples", delta["samples"][:12],
                "level", child.levels_completed,
            )


A.run_program("dc22", run)

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
TOP = (
    REMOTE
    + [2] * 8
    + [4, 4, A_CONTROL, 4, A_CONTROL, 1]
    + [A_CONTROL, 4] * 3
    + [1, 1, 1]
    + [S_CONTROL, S_CONTROL, 3, B_CONTROL]
)


def apply(env, path):
    for action in path:
        env.step(*action) if isinstance(action, tuple) else env.step(action)


def selector_region(env):
    frame = env.frame()
    return tuple(
        int(value)
        for value in frame[46:54, 16:30].ravel()
    )


def run(env):
    solver.solve(env)
    apply(env, TOP)
    for direction in (1, 2, 3, 4):
        child = env.clone()
        before = child.frame().copy()
        before_selector = selector_region(child)
        child.step(direction)
        delta = frame_delta(before[:63], child.frame()[:63])
        print(
            "ONE", direction,
            "delta", (delta["count"], delta["bbox"]),
            "selector_changed", selector_region(child) != before_selector,
            "samples", delta["samples"][:16],
        )
        repeated = child.clone()
        changes = []
        previous = selector_region(repeated)
        for turn in range(2, 10):
            repeated.step(direction)
            current = selector_region(repeated)
            if current != previous:
                changes.append(turn)
                previous = current
        print("REPEAT", direction, changes)


A.run_program("dc22", run)

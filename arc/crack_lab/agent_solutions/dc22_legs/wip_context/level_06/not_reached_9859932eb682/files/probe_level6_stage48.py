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
D_CONTROLS = {
    1: (6, 50, 32),
    2: (6, 50, 40),
    3: (6, 46, 36),
    4: (6, 54, 36),
}
REMOTE = [
    3, 3, 3, 3, 3,
    A_CONTROL, A_CONTROL, A_CONTROL, A_CONTROL,
    2, 2, 3, 3, 3, 2, 3, A_CONTROL, 1, A_CONTROL, 1, 1,
    B_CONTROL,
] + [1] * 17 + [3] + [2] * 11
HUB = (
    REMOTE
    + [2] * 8
    + [4, 4, A_CONTROL, 4, A_CONTROL, 1]
    + [A_CONTROL, 4] * 3
    + [1, 1, 1]
    + [S_CONTROL, S_CONTROL, S_CONTROL, 3, B_CONTROL]
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
    apply(env, HUB)
    for marker_direction in (1, 2, 3, 4):
        marker = env.clone()
        marker.step(marker_direction)
        effects = []
        for control_direction, control in D_CONTROLS.items():
            child = marker.clone()
            before = child.frame().copy()
            child.step(*control)
            delta = frame_delta(before[:63], child.frame()[:63])
            if delta["count"] or child.levels_completed > 5:
                effects.append((
                    control_direction,
                    (delta["count"], delta["bbox"]),
                    child.levels_completed,
                ))
        print("MARKER", marker_direction, avatar_tile(marker), "effects", effects)


A.run_program("dc22", run)

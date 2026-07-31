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
D_RIGHT = (6, 54, 36)
D_CONTROLS = {
    "up": (6, 50, 32),
    "left": (6, 46, 36),
    "right": D_RIGHT,
    "down": (6, 50, 40),
}
REMOTE = [
    3, 3, 3, 3, 3,
    A_CONTROL, A_CONTROL, A_CONTROL, A_CONTROL,
    2, 2, 3, 3, 3, 2, 3, A_CONTROL, 1, A_CONTROL, 1, 1,
    B_CONTROL,
] + [1] * 17 + [3] + [2] * 11
RIGHT_ARM = (
    REMOTE
    + [2] * 8
    + [4, 4, A_CONTROL, 4, A_CONTROL, 1]
    + [A_CONTROL, 4] * 3
    + [1, 1, 1]
    + [S_CONTROL, S_CONTROL, S_CONTROL, 3, B_CONTROL, 4, 4]
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


def assembly_columns(env):
    frame = env.frame()
    cols = []
    for row in range(14, 20):
        for col in range(20):
            if 8 in frame[2 * row:2 * row + 2, 2 * col:2 * col + 2]:
                cols.append(col)
    return None if not cols else (min(cols), max(cols))


def run(env):
    solver.solve(env)
    apply(env, RIGHT_ARM)
    print("ROOT", avatar_tile(env), "assembly", assembly_columns(env))
    for phase in range(1, 9):
        before = env.frame().copy()
        env.step(*D_RIGHT)
        delta = frame_delta(before[:63], env.frame()[:63])
        print(
            "SHIFT", phase, "avatar", avatar_tile(env),
            "assembly", assembly_columns(env),
            "delta", (delta["count"], delta["bbox"]),
            "level", env.levels_completed,
        )
        if phase == 1:
            revealed = env.clone()
            apply(revealed, [3, B_CONTROL, S_CONTROL, S_CONTROL, B_CONTROL])
            print(
                "PHASE1_AFTER_SHIFT", avatar_tile(revealed),
                "level", revealed.levels_completed,
            )
            for name, control in D_CONTROLS.items():
                child = env.clone()
                before_control = child.frame().copy()
                child.step(*control)
                control_delta = frame_delta(
                    before_control[:63], child.frame()[:63]
                )
                print(
                    "DOCKED_CONTROL", name,
                    (control_delta["count"], control_delta["bbox"]),
                    "assembly", assembly_columns(child),
                    "level", child.levels_completed,
                )
        if env.levels_completed > 5:
            break
        env.step(3)
        left = avatar_tile(env)
        env.step(4)
        print("REENTER", phase, left, avatar_tile(env))


A.run_program("dc22", run)

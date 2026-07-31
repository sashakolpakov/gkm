import importlib.util
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A
import gkm_legs as G


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
    for a_phase in range(6):
        for b_phase in range(2):
            hub = env.clone()
            for _ in range(a_phase):
                hub.step(*A_CONTROL)
            for _ in range(b_phase):
                hub.step(*B_CONTROL)
            apply(hub, [S_CONTROL, S_CONTROL, S_CONTROL, 3, B_CONTROL])
            for docked in (0, 1):
                root = hub.clone()
                if docked:
                    apply(root, [4, 4, D_RIGHT, 3])
                exits = []
                for direction in (1, 2, 3, 4):
                    child = root.clone()
                    child.step(direction)
                    child.step(direction)
                    exits.append((direction, avatar_tile(child), child.levels_completed))
                unusual = [
                    item for item in exits
                    if item[1] not in ((28, 17), (30, 17), (29, 16), (18, 27))
                ]
                if unusual or a_phase == 0 and b_phase == 0:
                    print(
                        "PHASES", (a_phase, b_phase, docked),
                        "root", avatar_tile(root), "exits", exits,
                    )


A.run_program("dc22", run)

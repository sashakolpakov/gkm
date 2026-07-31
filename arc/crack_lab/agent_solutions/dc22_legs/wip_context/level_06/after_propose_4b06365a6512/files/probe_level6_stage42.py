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
SHIFTED_SELECTOR = (
    REMOTE
    + [2] * 8
    + [4, 4, A_CONTROL, 4, A_CONTROL, 1]
    + [A_CONTROL, 4] * 3
    + [1, 1, 1]
    + [S_CONTROL, S_CONTROL, S_CONTROL, 3, B_CONTROL, 4, 4, D_RIGHT, 3, B_CONTROL]
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


def set_phase(env, current, target):
    for _ in range((target - current) % 4):
        env.step(*S_CONTROL)
    return target


def try_visit(root, target_phase, visit_path):
    child = root.clone()
    phase = set_phase(child, 3, target_phase)
    child.step(*B_CONTROL)
    apply(child, visit_path)
    visited = avatar_tile(child)
    outward = []
    for direction in (1, 2, 3, 4):
        edge = child.clone()
        edge.step(direction)
        outward.append((direction, avatar_tile(edge), edge.levels_completed))
    apply(child, reversed_path(visit_path))
    child.step(*B_CONTROL)
    phase = set_phase(child, phase, 1)
    child.step(*B_CONTROL)
    print(
        "TARGET", target_phase, "visited", visited,
        "outward", outward,
        "phase1", avatar_tile(child), "level", child.levels_completed,
    )


def reversed_path(path):
    opposite = {1: 2, 2: 1, 3: 4, 4: 3}
    return [opposite[action] for action in reversed(path)]


def run(env):
    solver.solve(env)
    apply(env, SHIFTED_SELECTOR)
    print("ROOT", avatar_tile(env))
    try_visit(env, 0, [1, 1, 4])
    try_visit(env, 0, [1, 1, 4, 4])
    try_visit(env, 2, [2, 2, 2, 4, 4, 4])


A.run_program("dc22", run)

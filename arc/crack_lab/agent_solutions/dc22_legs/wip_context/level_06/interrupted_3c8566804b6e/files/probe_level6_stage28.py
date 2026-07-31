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
    + [1, 1, 1, 3]
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


def visit(env, current, target):
    current = set_phase(env, current, target)
    env.step(*B_CONTROL)
    destination = avatar_tile(env)
    env.step(*B_CONTROL)
    returned = avatar_tile(env)
    return current, destination, returned


def try_missing(root, visits):
    child = root.clone()
    phase = 0
    trace = []
    for target in visits:
        phase, destination, returned = visit(child, phase, target)
        trace.append((target, destination, returned))
    phase = set_phase(child, phase, 1)
    child.step(*B_CONTROL)
    print(
        "VISITS", visits, "trace", trace,
        "MISSING", avatar_tile(child), "level", child.levels_completed,
    )


def run(env):
    solver.solve(env)
    apply(env, STAGED)
    for visits in ((0,), (2,), (3,), (0, 2, 3), (2, 0, 3), (3, 2, 0)):
        try_missing(env, visits)


A.run_program("dc22", run)

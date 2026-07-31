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
SELECTOR = (
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


def run(env):
    solver.solve(env)
    apply(env, SELECTOR)
    for endpoint_phase in (0, 2, 3):
        endpoint = env.clone()
        set_phase(endpoint, 0, endpoint_phase)
        endpoint.step(*B_CONTROL)
        destination = avatar_tile(endpoint)
        for selector_phase in range(4):
            child = endpoint.clone()
            set_phase(child, endpoint_phase, selector_phase)
            child.step(*B_CONTROL)
            print(
                "ENDPOINT", endpoint_phase, destination,
                "SELECTOR", selector_phase,
                "RESULT", avatar_tile(child), child.levels_completed,
            )


A.run_program("dc22", run)

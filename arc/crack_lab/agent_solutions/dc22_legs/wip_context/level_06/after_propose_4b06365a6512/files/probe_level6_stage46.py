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
    islands = {
        0: {(row, col) for row in range(24, 27) for col in range(16, 19)},
        2: {(row, col) for row in range(2, 6) for col in range(2, 6)},
    }
    for a_phase in range(6):
        for b_phase in range(2):
            aligned = env.clone()
            for _ in range(a_phase):
                aligned.step(*A_CONTROL)
            for _ in range(b_phase):
                aligned.step(*B_CONTROL)
            for endpoint_phase in (0, 2):
                endpoint = aligned.clone()
                for _ in range(endpoint_phase):
                    endpoint.step(*S_CONTROL)
                apply(endpoint, [3, B_CONTROL])
                exits = []
                for direction in (1, 2, 3, 4):
                    child = endpoint.clone()
                    child.step(direction)
                    child.step(direction)
                    result = avatar_tile(child)
                    if result not in islands[endpoint_phase]:
                        exits.append((direction, result, child.levels_completed))
                if exits:
                    print(
                        "PHASES", (a_phase, b_phase),
                        "ENDPOINT", endpoint_phase, avatar_tile(endpoint),
                        "EXITS", exits,
                    )


A.run_program("dc22", run)

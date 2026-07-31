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
    + [
        S_CONTROL, S_CONTROL, S_CONTROL, 3, B_CONTROL,
        4, 4, D_RIGHT, 3, B_CONTROL, 4,
    ]
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


def pair_tiles(env, wanted):
    frame = env.frame()
    out = []
    for row in range(31):
        for col in range(20):
            colors = tuple(sorted(set(int(v) for v in frame[
                2 * row:2 * row + 2, 2 * col:2 * col + 2
            ].ravel())))
            if colors == wanted:
                out.append((row, col))
    return out


def run(env):
    solver.solve(env)
    apply(env, STAGED)
    for a_phase in range(6):
        for b_phase in range(2):
            child = env.clone()
            for _ in range(a_phase):
                child.step(*A_CONTROL)
            for _ in range(b_phase):
                child.step(*B_CONTROL)
            pairs = pair_tiles(child, (8, 13))
            child.step(*S_CONTROL)
            child.step(*S_CONTROL)
            child.step(3)
            child.step(*B_CONTROL)
            print(
                "PHASES", (a_phase, b_phase), "pairs", pairs,
                "destination", avatar_tile(child), "level", child.levels_completed,
            )


A.run_program("dc22", run)

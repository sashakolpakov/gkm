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
LOWER_TRANSFER = (
    [2] * 8
    + [4, 4, A_CONTROL, 4, A_CONTROL, 1]
    + [A_CONTROL, 4] * 3
    + [1, 1, 1]
)
VISITS = {
    "none": ([], []),
    "right": ([1, 1, 1, 4, 4, B_CONTROL, 4, 4, 4], [3] * 5 + [2] * 3),
    "right_up": (
        [1, 1, 1, 4, 4, B_CONTROL, 4, 4, 4, 1],
        [2] + [3] * 5 + [2] * 3,
    ),
    "left": ([1, 1, 1, 3, B_CONTROL, 3, 3, 3], [4] * 4 + [2] * 3),
    "left_up": (
        [1, 1, 1, 3, B_CONTROL, 3, 3, 3, 1],
        [2] + [4] * 4 + [2] * 3,
    ),
}


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


def pair_tiles(env, wanted=(8, 13)):
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
    apply(env, REMOTE)
    for name, (outbound, inbound) in VISITS.items():
        child = env.clone()
        apply(child, outbound)
        visited = avatar_tile(child)
        revealed_at_visit = pair_tiles(child)
        apply(child, inbound + LOWER_TRANSFER)
        staged = avatar_tile(child)
        child.step(*S_CONTROL)
        child.step(3)
        child.step(*B_CONTROL)
        print(
            name, "visited", visited, "pairs", revealed_at_visit,
            "staged", staged, "destination", avatar_tile(child),
            "level", child.levels_completed,
        )


A.run_program("dc22", run)

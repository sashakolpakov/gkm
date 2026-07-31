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


def set_phase(env, current, target):
    for _ in range((target - current) % 4):
        env.step(*S_CONTROL)


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
    apply(env, RIGHT_ARM)
    visited = env.clone()
    apply(visited, [3, B_CONTROL, S_CONTROL, S_CONTROL, B_CONTROL])
    print(
        "VISIT_ONLY_PHASE1", avatar_tile(visited),
        pair_tiles(visited), visited.levels_completed,
    )
    for shift_phase in range(4):
        child = env.clone()
        set_phase(child, 3, shift_phase)
        before = child.frame().copy()
        child.step(*D_RIGHT)
        delta = frame_delta(before[:63], child.frame()[:63])
        after_shift = (avatar_tile(child), pair_tiles(child), child.levels_completed)
        set_phase(child, shift_phase, 3)
        apply(child, [3, B_CONTROL])
        set_phase(child, 3, 1)
        child.step(*B_CONTROL)
        print(
            "SHIFT_PHASE", shift_phase,
            "delta", (delta["count"], delta["bbox"]),
            "after", after_shift,
            "phase1", avatar_tile(child), child.levels_completed,
        )


A.run_program("dc22", run)

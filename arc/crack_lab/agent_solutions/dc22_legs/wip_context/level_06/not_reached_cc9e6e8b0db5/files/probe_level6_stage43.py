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
    + [S_CONTROL, S_CONTROL, S_CONTROL, 3, B_CONTROL, 4, 4, D_RIGHT]
)


def apply(env, path):
    for action in path:
        env.step(*action) if isinstance(action, tuple) else env.step(action)


def run(env):
    solver.solve(env)
    apply(env, RIGHT_ARM)
    active = []
    for a_phase in range(6):
        for b_phase in range(2):
            for s_phase in range(4):
                child = env.clone()
                for _ in range(a_phase):
                    child.step(*A_CONTROL)
                for _ in range(b_phase):
                    child.step(*B_CONTROL)
                for _ in range(s_phase):
                    child.step(*S_CONTROL)
                before = child.frame().copy()
                child.step(*D_RIGHT)
                delta = frame_delta(before[:63], child.frame()[:63])
                if delta["count"] or child.levels_completed > 5:
                    active.append((
                        (a_phase, b_phase, s_phase),
                        (delta["count"], delta["bbox"]),
                        child.levels_completed,
                    ))
    print("SECOND_SHIFTS", active)


A.run_program("dc22", run)

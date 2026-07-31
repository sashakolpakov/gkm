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

REMOTE = [
    3, 3, 3, 3, 3,
    (6, 56, 8), (6, 56, 8), (6, 56, 8), (6, 56, 8),
    2, 2, 3, 3, 3, 2, 3, (6, 56, 8), 1, (6, 56, 8), 1, 1,
    (6, 51, 25),
] + [1] * 17 + [3] + [2] * 11

A_CONTROL = (6, 56, 8)
B_CONTROL = (6, 51, 25)


def apply(env, path):
    for action in path:
        env.step(*action) if isinstance(action, tuple) else env.step(action)


def avatar_tile(env):
    cells = list(zip(*((env.frame() == 14).nonzero())))
    return None if not cells else (int(cells[0][0] // 2), int(cells[0][1] // 2))


def run(env):
    solver.solve(env)
    apply(env, REMOTE)
    exits = {}
    for a_phase in range(8):
        for b_phase in range(8):
            child = env.clone()
            for _ in range(a_phase):
                child.step(*A_CONTROL)
            for _ in range(b_phase):
                child.step(*B_CONTROL)
            child.step(1)
            result = (avatar_tile(child), child.levels_completed)
            exits.setdefault(result, []).append((a_phase, b_phase))
    for result, phases in exits.items():
        print("EXIT", result, "phases", phases)


A.run_program("dc22", run)

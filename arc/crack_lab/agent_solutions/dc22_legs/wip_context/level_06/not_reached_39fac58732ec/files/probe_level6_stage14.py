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

RIGHT_FRONTIER = [
    3, 3, 3, 3, 3,
    (6, 56, 8), (6, 56, 8), (6, 56, 8), (6, 56, 8),
    2, 2, 3, 3, 3, 2, 3, (6, 56, 8), 1, (6, 56, 8), 1, 1,
    (6, 51, 25),
] + [1] * 17 + [3] + [2] * 11 + [1, 1, 1, 4, 4, (6, 51, 25), 4, 4]


def apply(env, path):
    for action in path:
        env.step(*action) if isinstance(action, tuple) else env.step(action)


def avatar_tile(env):
    cells = list(zip(*((env.frame() == 14).nonzero())))
    return None if not cells else (int(cells[0][0] // 2), int(cells[0][1] // 2))


def run(env):
    solver.solve(env)
    apply(env, RIGHT_FRONTIER)
    print("ROOT", avatar_tile(env))
    phased = env.clone()
    for phase in range(7):
        child = phased.clone()
        transitions = []
        previous = avatar_tile(child)
        for turn in range(1, 25):
            child.step(4)
            current = avatar_tile(child)
            if current != previous or child.levels_completed > 5:
                transitions.append((turn, current, child.levels_completed))
                previous = current
            if child.levels_completed > 5:
                break
        print("PHASE", phase, "transitions", transitions)
        phased.step(6, 51, 25)


A.run_program("dc22", run)

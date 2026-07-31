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


def apply(env, path):
    for action in path:
        env.step(*action) if isinstance(action, tuple) else env.step(action)


def avatar_tile(env):
    cells = list(zip(*((env.frame() == 14).nonzero())))
    return None if not cells else (int(cells[0][0] // 2), int(cells[0][1] // 2))


def selector_pair(env):
    block = env.frame()[48:50, 18:20]
    return tuple(sorted(set(int(v) for v in block.ravel())))


def run(env):
    solver.solve(env)
    apply(env, REMOTE)
    print("REMOTE", avatar_tile(env), selector_pair(env))
    phased = env.clone()
    for phase in range(4):
        print("PHASE", phase, "pair", selector_pair(phased))
        for action in (1, 2, 3, 4):
            child = phased.clone()
            child.step(action)
            print(" MOVE", action, avatar_tile(child), "level", child.levels_completed)
        phased.step(6, 51, 48)


A.run_program("dc22", run)

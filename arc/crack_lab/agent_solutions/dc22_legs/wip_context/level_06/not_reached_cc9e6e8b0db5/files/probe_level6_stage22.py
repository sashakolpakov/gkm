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

S = (6, 51, 48)
ENTRY = [1, 1] + [3] * 5


def avatar_tile(env):
    frame = env.frame()
    for row in range(31):
        for col in range(32):
            if (frame[2 * row:2 * row + 2, 2 * col:2 * col + 2] == 14).all():
                return row, col
    return None


def run(env):
    solver.solve(env)
    phased = env.clone()
    for phase in range(4):
        child = phased.clone()
        trace = [(0, avatar_tile(child), child.levels_completed)]
        for turn, action in enumerate(ENTRY, 1):
            child.step(action)
            trace.append((turn, avatar_tile(child), child.levels_completed))
        exits = []
        for direction in (1, 2, 3, 4):
            grandchild = child.clone()
            grandchild.step(direction)
            exits.append((direction, avatar_tile(grandchild), grandchild.levels_completed))
        print("PHASE", phase, "trace", trace, "next", exits)
        phased.step(*S)


A.run_program("dc22", run)

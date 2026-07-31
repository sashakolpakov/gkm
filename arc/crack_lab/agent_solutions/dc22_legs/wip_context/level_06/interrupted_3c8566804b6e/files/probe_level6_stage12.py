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


def lower_patterns(env):
    frame = env.frame()
    out = []
    for r in range(44, 62, 2):
        for c in range(12, 40, 2):
            colors = tuple(sorted(set(int(v) for v in frame[r:r + 2, c:c + 2].ravel())))
            if len(colors) > 1:
                out.append((r // 2, c // 2, colors))
    return tuple(out)


def run(env):
    solver.solve(env)
    apply(env, REMOTE)
    phased = env.clone()
    for phase in range(4):
        child = phased.clone()
        previous = lower_patterns(child)
        transitions = [(0, previous)]
        for turn in range(1, 21):
            child.step(2)
            current = lower_patterns(child)
            if current != previous:
                transitions.append((turn, current))
                previous = current
        print("PHASE", phase, "transitions", transitions,
              "level", child.levels_completed)
        phased.step(6, 51, 48)


A.run_program("dc22", run)

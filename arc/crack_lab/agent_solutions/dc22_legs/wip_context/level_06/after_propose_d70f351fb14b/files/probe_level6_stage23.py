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

REMOTE = [
    3, 3, 3, 3, 3,
    (6, 56, 8), (6, 56, 8), (6, 56, 8), (6, 56, 8),
    2, 2, 3, 3, 3, 2, 3, (6, 56, 8), 1, (6, 56, 8), 1, 1,
    (6, 51, 25),
] + [1] * 17 + [3] + [2] * 11 + [1]

CONTROLS = {
    "A": (6, 56, 8),
    "B": (6, 51, 25),
    "S": (6, 51, 48),
}


def apply(env, path):
    for action in path:
        env.step(*action) if isinstance(action, tuple) else env.step(action)


def sweep(root, name, control):
    child = root.clone()
    seen = {child.frame()[:63].tobytes(): 0}
    summary = []
    for turn in range(1, 21):
        before = child.frame().copy()
        child.step(*control)
        delta = frame_delta(before[:63], child.frame()[:63])
        key = child.frame()[:63].tobytes()
        summary.append((turn, delta["count"], delta["bbox"], child.levels_completed))
        if key in seen:
            print(name, "cycle", (seen[key], turn), "steps", summary)
            return
        seen[key] = turn
    print(name, "no_cycle", len(seen), "steps", summary)


def run(env):
    solver.solve(env)
    initial = env.clone()
    returned = env.clone()
    apply(returned, REMOTE)
    for context, root in (("INITIAL", initial), ("RETURNED", returned)):
        for name, control in CONTROLS.items():
            sweep(root, context + "_" + name, control)


A.run_program("dc22", run)

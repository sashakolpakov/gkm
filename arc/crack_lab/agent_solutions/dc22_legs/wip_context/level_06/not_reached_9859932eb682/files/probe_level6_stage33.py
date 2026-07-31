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
    + [(6, 51, 48), (6, 51, 48), (6, 51, 48), 3, B_CONTROL, 4, 4]
)
D_PAD = {
    "up": (6, 50, 32),
    "left": (6, 46, 36),
    "center": (6, 50, 36),
    "right": (6, 54, 36),
    "down": (6, 50, 40),
}


def apply(env, path):
    for action in path:
        env.step(*action) if isinstance(action, tuple) else env.step(action)


def patterned_tiles(env):
    frame = env.frame()
    out = []
    for row in range(31):
        for col in range(20):
            colors = tuple(sorted(set(int(v) for v in frame[
                2 * row:2 * row + 2, 2 * col:2 * col + 2
            ].ravel())))
            if len(colors) > 1:
                out.append((row, col, colors))
    return out


def run(env):
    solver.solve(env)
    apply(env, STAGED)
    base = env.frame().copy()
    print("ROOT_PATTERNS", patterned_tiles(env))
    for name, control in D_PAD.items():
        child = env.clone()
        child.step(*control)
        delta = frame_delta(base[:63], child.frame()[:63])
        print(
            "ONE", name, "delta", (delta["count"], delta["bbox"]),
            "patterns", patterned_tiles(child),
            "level", child.levels_completed,
        )
        trace = []
        previous = child.frame()[:63].tobytes()
        for turn in range(2, 9):
            child.step(*control)
            current = child.frame()[:63].tobytes()
            if current != previous or child.levels_completed > 5:
                trace.append((turn, patterned_tiles(child), child.levels_completed))
                previous = current
        print("REPEAT", name, trace)


A.run_program("dc22", run)

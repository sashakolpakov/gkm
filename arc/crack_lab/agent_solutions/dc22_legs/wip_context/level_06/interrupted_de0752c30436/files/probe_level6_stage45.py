import importlib.util
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A
import gkm_legs as G

from perception import connected_components


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
SHIFTED_SELECTOR = (
    REMOTE
    + [2] * 8
    + [4, 4, A_CONTROL, 4, A_CONTROL, 1]
    + [A_CONTROL, 4] * 3
    + [1, 1, 1]
    + [
        S_CONTROL, S_CONTROL, S_CONTROL, 3, B_CONTROL,
        4, 4, D_RIGHT, 3, B_CONTROL, 4,
    ]
)


def apply(env, path):
    for action in path:
        env.step(*action) if isinstance(action, tuple) else env.step(action)


def components(env):
    return [
        (blob.bbox, blob.area)
        for blob in connected_components(env.frame(), colors=(8,), min_area=2)
        if blob.bbox[1] < 40
    ]


def hub_markers(env):
    frame = env.frame()
    out = []
    for row in range(28, 31):
        for col in range(16, 19):
            colors = tuple(sorted(set(int(v) for v in frame[
                2 * row:2 * row + 2, 2 * col:2 * col + 2
            ].ravel())))
            if any(color in colors for color in (10, 12, 14, 15)):
                out.append((row, col, colors))
    return out


def run(env):
    solver.solve(env)
    apply(env, SHIFTED_SELECTOR)
    phased = env.clone()
    for phase in range(4):
        print(
            "B_PHASE", phase, "components", components(phased),
            "hub", hub_markers(phased),
        )
        phased.step(*B_CONTROL)


A.run_program("dc22", run)

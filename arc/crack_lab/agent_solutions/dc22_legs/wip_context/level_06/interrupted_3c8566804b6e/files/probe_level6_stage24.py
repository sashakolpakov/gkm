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
S_CONTROL = (6, 51, 48)
LOWER_EDGE = (
    [2] * 8
    + [4, 4, A_CONTROL, 4, A_CONTROL, 1]
    + [A_CONTROL, 4] * 3
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


def transitions(root, direction, primed):
    child = root.clone()
    out = []
    previous = avatar_tile(child)
    for turn in range(1, 17):
        if primed:
            child.step(*A_CONTROL)
        child.step(direction)
        current = avatar_tile(child)
        if current != previous or child.levels_completed > 5:
            out.append((turn, current, child.levels_completed))
            previous = current
        if child.levels_completed > 5:
            break
    return out


def run(env):
    solver.solve(env)
    apply(env, REMOTE + LOWER_EDGE)
    print("ROOT", avatar_tile(env), "level", env.levels_completed)
    for direction in (1, 2, 3, 4):
        print(
            "DIR", direction,
            "plain", transitions(env, direction, False),
            "primed", transitions(env, direction, True),
        )
    staged = env.clone()
    apply(staged, [1, 1, 1])
    phased = staged.clone()
    for phase in range(4):
        child = phased.clone()
        child.step(3)
        print(
            "SELECTOR", phase, "from", avatar_tile(phased),
            "to", avatar_tile(child), "level", child.levels_completed,
        )
        phased.step(*S_CONTROL)
    right_edge = staged.clone()
    apply(right_edge, [2, 2, 4, 4, 4, 4])
    print(
        "RIGHT_EDGE", avatar_tile(right_edge),
        "plain", transitions(right_edge, 4, False),
        "primed", transitions(right_edge, 4, True),
    )
    occupied = staged.clone()
    occupied.step(3)
    trace = []
    for phase in range(1, 6):
        occupied.step(*S_CONTROL)
        trace.append((phase, avatar_tile(occupied), occupied.levels_completed))
    print("OCCUPIED_SELECTOR", trace)


A.run_program("dc22", run)

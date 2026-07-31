import importlib.util
import os
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A
import gkm_legs as G


taint_reason = G._workspace_taint_reason(os.getcwd())
if taint_reason:
    raise SystemExit(f"TAINTED WORKSPACE: {taint_reason}")

spec = importlib.util.spec_from_file_location("solve", "solve.py")
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)

A_CONTROL = (6, 56, 8)
B_CONTROL = (6, 51, 25)
S_CONTROL = (6, 51, 48)
D_LEFT = (6, 46, 36)
D_CONTROLS = {
    "u": (1, (6, 50, 32), 2),
    "d": (2, (6, 50, 40), 1),
    "l": (3, D_LEFT, 4),
    "r": (4, (6, 54, 36), 3),
}

REMOTE = [
    3, 3, 3, 3, 3,
    A_CONTROL, A_CONTROL, A_CONTROL, A_CONTROL,
    2, 2, 3, 3, 3, 2, 3, A_CONTROL, 1, A_CONTROL, 1, 1,
    B_CONTROL,
] + [1] * 17 + [3] + [2] * 11

HUB = (
    REMOTE
    + [2] * 8
    + [4, 4, A_CONTROL, 4, A_CONTROL, 1]
    + [A_CONTROL, 4] * 3
    + [1, 1, 1]
    + [S_CONTROL, S_CONTROL, S_CONTROL, 3, B_CONTROL]
)

REVERSE = [
    B_CONTROL, 4,
    2, 2, 2, 3, 2, 3, 3, 2, 3, 3, 1, A_CONTROL, 1,
] + [1] * 7 + [3]

ATTEMPTS = {
    "plain": [1] * 3 + [4] * 12,
    "b_at_edge": [1] * 3 + [B_CONTROL] + [4] * 12,
    "b_before": [B_CONTROL] + [1] * 3 + [4] * 12,
    "b_mid": [1] * 3 + [4, 4, B_CONTROL] + [4] * 10,
    "upper": [1] * 3 + [B_CONTROL, 4, 1] + [4] * 11,
    "lower": [1] * 3 + [B_CONTROL, 4, 2] + [4] * 11,
}

CENTRAL_ATTEMPTS = {
    "straight_up": [1] * 3 + [4] * 2 + [B_CONTROL] + [4] * 4 + [1] * 20,
    "in_up": (
        [1] * 3 + [4] * 2 + [B_CONTROL] + [4] * 3
        + [1, 4] + [1] * 20 + [4] * 12
    ),
    "in_up2": (
        [1] * 3 + [4] * 2 + [B_CONTROL] + [4] * 3
        + [1, 4, 4] + [1] * 20 + [4] * 12
    ),
    "in_down": (
        [1] * 3 + [4] * 2 + [B_CONTROL] + [4] * 3
        + [2, 4, 4] + [1] * 20 + [4] * 12
    ),
}

CONFIGS = {
    "u": ["u"],
    "ul": ["u", "l"],
    "ur": ["u", "r"],
    "uru": ["u", "r", "u"],
    "uruu": ["u", "r", "u", "u"],
    "uruul": ["u", "r", "u", "u", "l"],
    "uruull": ["u", "r", "u", "u", "l", "l"],
}


def step(env, action):
    env.step(*action) if isinstance(action, tuple) else env.step(action)


def apply(env, path):
    for action in path:
        step(env, action)


def avatar(env):
    rows, cols = np.where(env.frame()[:62, :40] == 14)
    if not len(rows):
        return None
    return (
        int(rows.min() // 2), int(cols.min() // 2),
        int(rows.max() // 2), int(cols.max() // 2), int(len(rows)),
    )


def run(env):
    solver.solve(env)
    apply(env, HUB)
    for left_count in (0, 1, 2, 3, 4):
        configured = env.clone()
        for _ in range(left_count):
            apply(configured, [3, D_LEFT, 4])
        apply(configured, REVERSE)
        print("ROOT", left_count, avatar(configured), flush=True)
        for name, path in ATTEMPTS.items():
            child = configured.clone()
            trace = []
            for index, action in enumerate(path, 1):
                step(child, action)
                current = avatar(child)
                if (
                    current is None
                    or current[1] >= 10
                    or child.levels_completed > 5
                ):
                    trace.append((index, current, child.levels_completed))
            print(
                "TRY", left_count, name, "final", avatar(child),
                "level", child.levels_completed, "trace", trace,
                flush=True,
            )
    for name, labels in CONFIGS.items():
        configured = env.clone()
        for label in labels:
            outward, control, inward = D_CONTROLS[label]
            apply(configured, [outward, control, inward])
        apply(configured, REVERSE)
        for attempt, path in CENTRAL_ATTEMPTS.items():
            child = configured.clone()
            apply(child, path)
            print(
                "CENTRAL", name, attempt, "final", avatar(child),
                "level", child.levels_completed, flush=True,
            )


A.run_program("dc22", run)

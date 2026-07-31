import importlib.util
import os
import sys
from collections import deque

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
MACROS = [
    ("u", (1,)), ("d", (2,)), ("l", (3,)), ("r", (4,)),
    ("a", (A_CONTROL,)), ("b", (B_CONTROL,)), ("s", (S_CONTROL,)),
    ("du", (1, (6, 50, 32), 2)),
    ("dd", (2, (6, 50, 40), 1)),
    ("dl", (3, (6, 46, 36), 4)),
    ("dr", (4, (6, 54, 36), 3)),
]


def apply(env, path):
    for action in path:
        env.step(*action) if isinstance(action, tuple) else env.step(action)


def run(env):
    solver.solve(env)
    apply(env, HUB)
    root = env.clone()

    def reconstruct(labels):
        child = root.clone()
        for label in labels:
            actions = dict(MACROS)[label]
            apply(child, actions)
        return child

    queue = deque([[]])
    seen = {root.frame()[:63].tobytes()}
    while queue and len(seen) < 5000:
        path = queue.popleft()
        if len(path) >= 35:
            continue
        for label, actions in MACROS:
            child_path = path + [label]
            child = reconstruct(path)
            apply(child, actions)
            if child.levels_completed > 5:
                print("WIN", child_path, "states", len(seen), flush=True)
                return
            key = child.frame()[:63].tobytes()
            if key in seen:
                continue
            seen.add(key)
            queue.append(child_path)
        if len(seen) % 250 < 11:
            print(
                "PROGRESS", len(seen), "queue", len(queue),
                "depth", len(path), flush=True,
            )
    print("DONE", len(seen), "queue", len(queue), flush=True)


levels, path, error = A.run_program("dc22", run)
print("HARNESS", levels, len(path), error)

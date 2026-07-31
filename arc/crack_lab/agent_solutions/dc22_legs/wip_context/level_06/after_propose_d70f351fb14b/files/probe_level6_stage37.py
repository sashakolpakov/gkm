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
D_UP = (6, 50, 32)
D_LEFT = (6, 46, 36)
D_RIGHT = (6, 54, 36)
D_DOWN = (6, 50, 40)
REMOTE = [
    3, 3, 3, 3, 3,
    A_CONTROL, A_CONTROL, A_CONTROL, A_CONTROL,
    2, 2, 3, 3, 3, 2, 3, A_CONTROL, 1, A_CONTROL, 1, 1,
    B_CONTROL,
] + [1] * 17 + [3] + [2] * 11
SHIFTED = (
    REMOTE
    + [2] * 8
    + [4, 4, A_CONTROL, 4, A_CONTROL, 1]
    + [A_CONTROL, 4] * 3
    + [1, 1, 1]
    + [S_CONTROL, S_CONTROL, S_CONTROL, 3, B_CONTROL, 4, 4, D_RIGHT]
)
ACTIONS = [
    ("u", 1), ("d", 2), ("l", 3), ("r", 4),
    ("b", B_CONTROL), ("s", S_CONTROL), ("a", A_CONTROL),
    ("du", D_UP), ("dl", D_LEFT), ("dr", D_RIGHT), ("dd", D_DOWN),
]


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


def run(env):
    solver.solve(env)
    apply(env, SHIFTED)
    print("ROOT", avatar_tile(env), "level", env.levels_completed, flush=True)
    queue = deque([(env.clone(), [])])
    seen = {env.frame()[:63].tobytes()}
    positions = {avatar_tile(env)}
    while queue and len(seen) < 5000:
        node, path = queue.popleft()
        if len(path) >= 50:
            continue
        for label, action in ACTIONS:
            child = node.clone()
            child.step(*action) if isinstance(action, tuple) else child.step(action)
            child_path = path + [label]
            if child.levels_completed > 5:
                print("WIN", child_path, "states", len(seen), flush=True)
                return
            key = child.frame()[:63].tobytes()
            if key in seen:
                continue
            seen.add(key)
            position = avatar_tile(child)
            if position not in positions:
                positions.add(position)
                print("NEW", position, child_path, flush=True)
            queue.append((child, child_path))
    print("DONE", len(seen), "queue", len(queue), "positions", sorted(positions), flush=True)


A.run_program("dc22", run)

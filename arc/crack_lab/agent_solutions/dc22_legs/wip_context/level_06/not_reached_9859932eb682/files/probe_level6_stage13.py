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

REMOTE = [
    3, 3, 3, 3, 3,
    (6, 56, 8), (6, 56, 8), (6, 56, 8), (6, 56, 8),
    2, 2, 3, 3, 3, 2, 3, (6, 56, 8), 1, (6, 56, 8), 1, 1,
    (6, 51, 25),
] + [1] * 17 + [3] + [2] * 11

ACTIONS = [
    ("u", 1), ("d", 2), ("l", 3), ("r", 4),
    ("a", (6, 56, 8)), ("b", (6, 51, 25)), ("s", (6, 51, 48)),
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


def key(env):
    return env.frame()[:63].tobytes()


def run(env):
    solver.solve(env)
    apply(env, REMOTE)
    queue = deque([(env.clone(), [])])
    seen = {key(env)}
    avatar_positions = {avatar_tile(env)}
    while queue and len(seen) < 3000:
        node, path = queue.popleft()
        if len(path) >= 50:
            continue
        for label, action in ACTIONS:
            child = node.clone()
            child.step(*action) if isinstance(action, tuple) else child.step(action)
            child_path = path + [label]
            if child.levels_completed > 5:
                print("WIN", child_path, flush=True)
                return
            child_key = key(child)
            if child_key in seen:
                continue
            seen.add(child_key)
            position = avatar_tile(child)
            if position not in avatar_positions:
                avatar_positions.add(position)
                print("NEW_AVATAR", position, child_path, flush=True)
            queue.append((child, child_path))
        if len(seen) % 100 < 7:
            print("PROGRESS", len(seen), "queue", len(queue), flush=True)
    print("DONE", len(seen), "queue", len(queue),
          "positions", sorted(avatar_positions), flush=True)


A.run_program("dc22", run)

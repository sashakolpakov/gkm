import importlib.util
import sys
from collections import deque

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena


spec = importlib.util.spec_from_file_location("solve", "solve.py")
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)

A = (6, 56, 8)
B = (6, 51, 25)
S = (6, 51, 48)
D = {
    "U": (1, (6, 50, 32), 2),
    "D": (2, (6, 50, 40), 1),
    "L": (3, (6, 46, 36), 4),
    "R": (4, (6, 54, 36), 3),
}
REMOTE = (
    [3] * 5
    + [A] * 4
    + [2, 2, 3, 3, 3, 2, 3, A, 1, A, 1, 1, B]
    + [1] * 17
    + [3]
    + [2] * 11
)
ROOT_TO_SELECTOR = (
    [2] * 8
    + [4, 4, A, 4, A, 1]
    + [A, 4] * 3
    + [1, 1, 1]
)
HUB = REMOTE + ROOT_TO_SELECTOR + [S, S, S, 3, B]
MACROS = (
    ("A", (A,)), ("B", (B,)), ("S", (S,)),
    ("U", D["U"]), ("D", D["D"]), ("L", D["L"]), ("R", D["R"]),
)


def step(env, action):
    env.step(*action) if isinstance(action, tuple) else env.step(action)


def apply(env, actions):
    for action in actions:
        step(env, action)


def key(env):
    return np.asarray(env.frame())[:62, :40].tobytes()


def avatar(env):
    rows, cols = np.where(np.asarray(env.frame())[:62, :40] == 14)
    return None if not len(rows) else (int(rows.min() // 2), int(cols.min() // 2))


def goal_metric(env):
    frame = np.asarray(env.frame())
    best = 0
    solid = []
    total = 0
    for row in range(31):
        for col in range(20):
            block = frame[2 * row:2 * row + 2, 2 * col:2 * col + 2]
            count = int(np.count_nonzero(block == 11))
            best = max(best, count)
            total += count
            if count == 4:
                solid.append((row, col))
    return best, total, solid


def run(env):
    solver.solve(env)
    apply(env, HUB)
    queue = deque([(env.clone(), [])])
    seen = {key(env)}
    best_metric = goal_metric(env)[:2]
    print("ROOT", best_metric, flush=True)
    while queue and len(seen) < 10000:
        node, path = queue.popleft()
        for label, macro in MACROS:
            if label in D and avatar(node) != (29, 17):
                continue
            child = node.clone()
            apply(child, macro)
            child_path = path + [label]
            metric = goal_metric(child)
            if metric[2]:
                print(
                    "GOAL", metric, "path", child_path,
                    "level", child.levels_completed,
                    "states", len(seen), flush=True,
                )
                return
            if metric[:2] > best_metric:
                best_metric = metric[:2]
                print(
                    "PROGRESS", best_metric, "path", child_path,
                    "states", len(seen), flush=True,
                )
            child_key = key(child)
            if child_key in seen:
                continue
            seen.add(child_key)
            queue.append((child, child_path))
        if len(seen) % 250 < len(MACROS):
            print(
                "STATES", len(seen), "queue", len(queue),
                "best", best_metric, flush=True,
            )
    print(
        "DONE", len(seen), "queue", len(queue),
        "best", best_metric, flush=True,
    )


arena.run_program("dc22", run)

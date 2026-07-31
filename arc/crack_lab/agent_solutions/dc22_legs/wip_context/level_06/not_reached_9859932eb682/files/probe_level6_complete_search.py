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
TOP_RING = ["U", "R", "U", "U", "L", "L", "U", "U", "U"]
MACROS = (
    ("u", (1,)), ("d", (2,)), ("l", (3,)), ("r", (4,)),
    ("a", (A,)), ("b", (B,)), ("s", (S,)),
    ("U", D["U"]), ("D", D["D"]), ("L", D["L"]), ("R", D["R"]),
)


def step(env, action):
    env.step(*action) if isinstance(action, tuple) else env.step(action)


def apply(env, actions):
    for action in actions:
        step(env, action)


def world_key(env):
    return np.asarray(env.frame())[:62, :40].tobytes()


def avatar(env):
    rows, cols = np.where(np.asarray(env.frame())[:62, :40] == 14)
    return None if not len(rows) else (int(rows.min() // 2), int(cols.min() // 2))


def run(env):
    solver.solve(env)
    apply(env, HUB)
    print("ROOT", avatar(env), env.levels_completed, flush=True)

    queue = deque([(env.clone(), [], [])])
    seen = {world_key(env)}
    positions = {avatar(env)}
    best = (avatar(env)[0], avatar(env)[1], 1)
    base_level = env.levels_completed

    while queue and len(seen) < 10000:
        node, labels, actions = queue.popleft()
        for label, macro in MACROS:
            child_actions = actions + list(macro)
            if len(child_actions) > 170:
                continue
            child = node.clone()
            apply(child, macro)
            child_labels = labels + [label]
            if child.levels_completed > base_level:
                print(
                    "WIN", child_labels, "actions", child_actions,
                    "states", len(seen), flush=True,
                )
                return
            key = world_key(child)
            if key in seen:
                continue
            seen.add(key)
            position = avatar(child)
            if position not in positions:
                positions.add(position)
                visible = [item for item in positions if item is not None]
                metric = (
                    min(row for row, _ in visible),
                    max(col for _, col in visible),
                    len(visible),
                )
                if (
                    metric[0] < best[0]
                    or metric[1] > best[1]
                    or metric[2] > best[2]
                ):
                    best = (
                        min(best[0], metric[0]),
                        max(best[1], metric[1]),
                        max(best[2], metric[2]),
                    )
                    print(
                        "FRONTIER", position, "metric", metric,
                        "path_len", len(child_actions),
                        "labels", child_labels, flush=True,
                    )
            queue.append((child, child_labels, child_actions))
        if len(seen) % 500 < len(MACROS):
            print(
                "STATES", len(seen), "queue", len(queue),
                "best", best, flush=True,
            )
    print(
        "DONE", len(seen), "queue", len(queue),
        "positions", len(positions), "best", best, flush=True,
    )


arena.run_program("dc22", run)

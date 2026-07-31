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
ACTIONS = (
    ("u", 1), ("d", 2), ("l", 3), ("r", 4),
    ("a", A), ("b", B), ("s", S),
)


def step(env, action):
    env.step(*action) if isinstance(action, tuple) else env.step(action)


def apply(env, actions):
    for action in actions:
        step(env, action)


def key(env):
    return np.asarray(env.frame())[:62, :40].tobytes()


def solid_goals(env):
    frame = np.asarray(env.frame())
    return [
        (row, col)
        for row in range(31)
        for col in range(20)
        if np.all(frame[
            2 * row:2 * row + 2,
            2 * col:2 * col + 2,
        ] == 11)
    ]


def run(env):
    solver.solve(env)
    apply(env, HUB)
    base_level = env.levels_completed
    queue = deque([(env.clone(), [], [])])
    seen = {key(env)}
    max_depth = 0
    while queue and len(seen) < 12000:
        node, labels, actions = queue.popleft()
        max_depth = max(max_depth, len(actions))
        for label, action in ACTIONS:
            child = node.clone()
            step(child, action)
            child_labels = labels + [label]
            child_actions = actions + [action]
            if child.levels_completed > base_level:
                print(
                    "WIN", child_labels, "actions", child_actions,
                    "states", len(seen), flush=True,
                )
                return
            goals = solid_goals(child)
            if goals:
                print(
                    "GOAL", goals, "labels", child_labels,
                    "actions", child_actions, "states", len(seen), flush=True,
                )
                return
            child_key = key(child)
            if child_key in seen or len(child_actions) >= 170:
                continue
            seen.add(child_key)
            queue.append((child, child_labels, child_actions))
        if len(seen) % 500 < len(ACTIONS):
            print(
                "STATES", len(seen), "queue", len(queue),
                "depth", max_depth, flush=True,
            )
    print(
        "DONE", len(seen), "queue", len(queue),
        "depth", max_depth, flush=True,
    )


arena.run_program("dc22", run)

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


def step(env, action):
    env.step(*action) if isinstance(action, tuple) else env.step(action)


def apply(env, actions):
    for action in actions:
        step(env, action)


def avatar(env):
    rows, cols = np.where(np.asarray(env.frame())[:62, :40] == 14)
    return None if not len(rows) else (int(rows.min() // 2), int(cols.min() // 2))


def movement_nodes(root):
    nodes = {avatar(root): (root.clone(), [])}
    queue = deque(nodes)
    while queue:
        position = queue.popleft()
        node, path = nodes[position]
        for direction in (1, 2, 3, 4):
            child = node.clone()
            child.step(direction)
            next_position = avatar(child)
            if next_position not in nodes:
                nodes[next_position] = (child, path + [direction])
                queue.append(next_position)
    return nodes


def tile(env, position):
    row, col = position
    block = np.asarray(env.frame())[
        2 * row:2 * row + 2,
        2 * col:2 * col + 2,
    ]
    return tuple(int(value) for value in block.ravel())


def solid_goals(env):
    frame = np.asarray(env.frame())
    return tuple(
        (row, col)
        for row in range(31)
        for col in range(20)
        if np.all(frame[
            2 * row:2 * row + 2,
            2 * col:2 * col + 2,
        ] == 11)
    )


def non_avatar_count(before, after):
    changed = before != after
    old = before[changed]
    new = after[changed]
    return int(np.count_nonzero((old != 14) & (new != 14)))


def classify(label, root):
    nodes = movement_nodes(root)
    groups = {}
    for position, (node, path) in nodes.items():
        before = np.asarray(node.frame())[:62, :40].copy()
        child = node.clone()
        step(child, B)
        signature = (
            non_avatar_count(
                before, np.asarray(child.frame())[:62, :40]
            ),
            avatar(child),
            tile(child, (2, 2)),
            solid_goals(child),
            child.levels_completed,
        )
        groups.setdefault(signature, []).append((position, path))
    print(
        label, "reach", sorted(nodes), "groups", groups, flush=True,
    )


def run(env):
    solver.solve(env)
    apply(env, REMOTE + ROOT_TO_SELECTOR)
    selector = env.clone()
    for phase, label in enumerate(("LOWER", "START", "TOP", "HUB")):
        endpoint = selector.clone()
        apply(endpoint, [S] * phase + [3, B])
        classify(label, endpoint)


arena.run_program("dc22", run)

"""Reproduce and inspect the shifted-ring physical opening."""
from collections import deque
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

import perception
import solve


A = (6, 56, 8)
B = (6, 50, 26)
S = (6, 50, 46)
DL = (6, 46, 36)
CONTROLS = {
    "A": A,
    "B": B,
    "S": S,
    "DU": (6, 50, 34),
    "DD": (6, 50, 40),
    "DL": DL,
    "DR": (6, 54, 36),
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
SHIFTED_SELECTOR = HUB + [3, DL, 4, 3, DL, 4, B]
REVERSE = (
    [4, 2, 2, 2, 3, 2, 3, 3, 2, 3, 3, 1, A, 1]
    + [1] * 7
    + [3]
)
OPEN = SHIFTED_SELECTOR + REVERSE + [1, 1, 1, 4, 4, B]


def step(env, action):
    env.step(*action) if isinstance(action, tuple) else env.step(action)


def apply(env, path):
    for action in path:
        step(env, action)


def avatar(env):
    frame = np.asarray(env.frame())
    for row in range(0, 62, 2):
        for col in range(0, 40, 2):
            if np.all(frame[row:row + 2, col:col + 2] == 14):
                return row, col
    return None


def frame_key(env):
    return np.asarray(env.frame())[:62, :40].tobytes()


def closure(root):
    queue = deque([(root.clone(), [])])
    nodes = {avatar(root): (root.clone(), [])}
    win = None
    while queue:
        node, path = queue.popleft()
        for direction in (1, 2, 3, 4):
            child = node.clone()
            child.step(direction)
            child_path = path + [direction]
            if child.levels_completed > root.levels_completed:
                win = child_path
                return nodes, win
            position = avatar(child)
            if position not in nodes:
                nodes[position] = (child, child_path)
                queue.append((child, child_path))
    return nodes, win


def ring(env):
    return tuple(
        (blob.bbox, blob.area)
        for blob in perception.connected_components(
            env.frame(), colors=(8,), min_area=20
        )
        if blob.bbox[1] < 40
    )


def solid_exits(env):
    frame = np.asarray(env.frame())
    return tuple(
        (row, col)
        for row in range(0, 62, 2)
        for col in range(0, 40, 2)
        if np.all(frame[row:row + 2, col:col + 2] == 11)
    )


def observe(env):
    solve.solve(env)
    base = env.clone()
    apply(base, REMOTE)
    base_nodes, _ = closure(base)

    apply(env, OPEN)
    nodes, win = closure(env)
    new_positions = sorted(set(nodes) - set(base_nodes), key=repr)
    print(
        "OPENED", "suffix", len(OPEN), "avatar", avatar(env),
        "ring", ring(env), "reach", len(nodes),
        "new", new_positions, "win", win,
        "exits", solid_exits(env), flush=True,
    )

    for position in new_positions:
        node, path = nodes[position]
        effects = []
        for name, control in CONTROLS.items():
            child = node.clone()
            before = frame_key(child)
            step(child, control)
            after = frame_key(child)
            changed = sum(a != b for a, b in zip(before, after))
            if (
                name.startswith("D") and changed
                or avatar(child) != position
                or child.levels_completed > env.levels_completed
                or solid_exits(child)
            ):
                effects.append(
                    (
                        name, changed, avatar(child), ring(child),
                        solid_exits(child), child.levels_completed,
                    )
                )
        if effects:
            print(
                "OPENED_CONTEXT", position, "walk", path,
                "effects", effects, flush=True,
            )


arena.run_program("dc22", observe)

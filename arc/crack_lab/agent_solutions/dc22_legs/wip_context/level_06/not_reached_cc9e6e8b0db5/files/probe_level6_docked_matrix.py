import importlib.util
import sys
from collections import deque

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from perception import connected_components


spec = importlib.util.spec_from_file_location("solve", "solve.py")
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)

A = (6, 56, 8)
B = (6, 51, 25)
S = (6, 51, 48)
D_LEFT = (6, 46, 36)
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
REVERSE_TO_ROOT = (
    [4, 2, 2, 2, 3, 2, 3, 3, 2, 3, 3, 1, A, 1]
    + [1] * 7
    + [3]
)


def step(env, action):
    env.step(*action) if isinstance(action, tuple) else env.step(action)


def apply(env, actions):
    for action in actions:
        step(env, action)


def avatar(env):
    rows, cols = np.where(np.asarray(env.frame())[:62, :40] == 14)
    return None if not len(rows) else (int(rows.min() // 2), int(cols.min() // 2))


def ring(env):
    blobs = [
        blob for blob in connected_components(
            env.frame(), colors=(8,), min_area=20
        )
        if blob.bbox[1] < 40
    ]
    if not blobs:
        return None
    blob = max(blobs, key=lambda item: item.area)
    return blob.bbox, blob.area


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


def run(env):
    solver.solve(env)
    apply(env, HUB)
    apply(env, [3, D_LEFT, 4] * 3)
    apply(env, [B] + REVERSE_TO_ROOT + [1] * 5)
    apply(env, [B] + [2] * 5 + ROOT_TO_SELECTOR + [3, B])
    nodes = movement_nodes(env)
    print(
        "DOCKED", avatar(env), ring(env), "reach", sorted(nodes),
        env.levels_completed, flush=True,
    )

    for family in ("SB", "BSB"):
        groups = {}
        for selector_phase in range(4):
            sequence = (
                [S] * selector_phase + [B]
                if family == "SB"
                else [B] + [S] * selector_phase + [B]
            )
            for position, (node, path) in nodes.items():
                child = node.clone()
                apply(child, sequence)
                signature = (
                    selector_phase,
                    avatar(child),
                    ring(child),
                    child.levels_completed,
                )
                groups.setdefault(signature, []).append((position, path))
        for signature, entries in groups.items():
            print(
                family, signature,
                "positions", [position for position, _ in entries],
                "example", entries[0], flush=True,
            )


arena.run_program("dc22", run)

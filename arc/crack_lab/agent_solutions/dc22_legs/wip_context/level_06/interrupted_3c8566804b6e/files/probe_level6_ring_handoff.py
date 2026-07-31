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
D = {
    "u": (1, (6, 50, 32), 2),
    "d": (2, (6, 50, 40), 1),
    "l": (3, (6, 46, 36), 4),
    "r": (4, (6, 54, 36), 3),
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
REVERSE_TO_ROOT = (
    [4, 2, 2, 2, 3, 2, 3, 3, 2, 3, 3, 1, A, 1]
    + [1] * 7
    + [3]
)
RING_PATHS = [
    [],
    ["u"], ["d"], ["l"], ["r"],
    ["u", "l"], ["u", "r"], ["d", "l"], ["d", "r"],
    ["l", "l"], ["u", "r", "u"], ["l", "l", "l"],
    ["u", "r", "u", "u"], ["l", "l", "l", "l"],
    ["u", "r", "u", "u", "l"],
    ["u", "r", "u", "u", "l", "l"],
    ["u", "r", "u", "u", "l", "l", "u"],
    ["u", "r", "u", "u", "l", "l", "u", "u"],
    ["u", "r", "u", "u", "l", "l", "u", "u", "u"],
]


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
    return None if not blobs else max(blobs, key=lambda blob: blob.area).bbox


def move_ring(env, labels):
    for label in labels:
        outward, control, inward = D[label]
        apply(env, [outward, control, inward])


def movement_reach(root):
    nodes = {avatar(root): root.clone()}
    queue = deque(nodes)
    win = None
    while queue and len(nodes) < 140:
        position = queue.popleft()
        node = nodes[position]
        for direction in (1, 2, 3, 4):
            child = node.clone()
            child.step(direction)
            next_position = avatar(child)
            if child.levels_completed > 5:
                win = (position, direction)
                return nodes, win
            if next_position not in nodes:
                nodes[next_position] = child
                queue.append(next_position)
    return nodes, win


def metric(nodes):
    positions = [position for position in nodes if position is not None]
    return (
        len(positions),
        min(row for row, _ in positions),
        max(row for row, _ in positions),
        min(col for _, col in positions),
        max(col for _, col in positions),
    )


def run(env):
    solver.solve(env)
    apply(env, HUB)
    for labels in RING_PATHS:
        configured = env.clone()
        move_ring(configured, labels)
        ring_box = ring(configured)
        apply(configured, [B] + REVERSE_TO_ROOT + [1, 1, 1, 4, 4])
        if avatar(configured) != (17, 6):
            print("BAD_STAGE", labels, avatar(configured), ring_box, flush=True)
            continue
        results = []
        phased = configured.clone()
        for phase in range(2):
            nodes, win = movement_reach(phased)
            results.append((phase, metric(nodes), win))
            step(phased, B)
        print("CONFIG", labels, "ring", ring_box, "results", results, flush=True)
        if any(result[2] is not None for result in results):
            return


arena.run_program("dc22", run)

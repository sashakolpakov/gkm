"""Visit every ring placement, then test the accumulated physical world."""
from collections import deque
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

import perception
import solve
from probe_l6_right import enter_right


A = (6, 56, 8)
B = (6, 50, 26)
S = (6, 50, 46)
MOVEMENT = {"U": 1, "D": 2, "L": 3, "R": 4}
INVERSE = {"U": 2, "D": 1, "L": 4, "R": 3}
CONTROL = {
    "U": (6, 50, 34),
    "D": (6, 50, 40),
    "L": (6, 46, 36),
    "R": (6, 54, 36),
}
ROUTE = "RL" + "DLRRLU" + "LLLLRRRR" + "URUULLUUU" + "DDDRRDDLD"
REVERSE_TO_PHYSICAL = (
    [4, 2, 2, 2, 3, 2, 3, 3, 2, 3, 3, 1, A, 1]
    + [1] * 7
    + [3]
)


def avatar(env):
    frame = np.asarray(env.frame())
    for row in range(0, 62, 2):
        for col in range(0, 40, 2):
            if np.all(frame[row:row + 2, col:col + 2] == 14):
                return row, col
    return None


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


def closure(root):
    nodes = {avatar(root): []}
    queue = deque([(root.clone(), [])])
    while queue:
        node, path = queue.popleft()
        for direction in (1, 2, 3, 4):
            child = node.clone()
            child.step(direction)
            child_path = path + [direction]
            if child.levels_completed > root.levels_completed:
                return nodes, child_path
            position = avatar(child)
            if position not in nodes:
                nodes[position] = child_path
                queue.append((child, child_path))
    return nodes, None


def observe(env):
    solve.solve(env)
    node = enter_right(env, 3)
    current = None
    ring_actions = 0
    for target in ROUTE:
        if current is not None:
            node.step(INVERSE[current])
            ring_actions += 1
        node.step(MOVEMENT[target])
        node.step(*CONTROL[target])
        ring_actions += 2
        current = target
    print(
        "ALL_RING", len(ROUTE), ring_actions,
        "avatar", avatar(node), "ring", ring(node),
        "exits", solid_exits(node), "level", node.levels_completed,
        flush=True,
    )

    node.step(INVERSE[current])
    node.step(*B)
    for action in REVERSE_TO_PHYSICAL:
        node.step(*action) if isinstance(action, tuple) else node.step(action)
    positions, win = closure(node)
    print(
        "ALL_RING_PHYSICAL", "avatar", avatar(node),
        "reach", len(positions), "bounds",
        (min(positions, key=repr), max(positions, key=repr)),
        "win", win, "ring", ring(node),
        "exits", solid_exits(node), "level", node.levels_completed,
        flush=True,
    )
    for b_phase in range(2):
        for a_phase in range(6):
            branch = node.clone()
            for _ in range(b_phase):
                branch.step(*B)
            for _ in range(a_phase):
                branch.step(*A)
            reached, branch_win = closure(branch)
            if (
                branch_win
                or solid_exits(branch)
                or len(reached) > len(positions)
            ):
                print(
                    "ALL_RING_PHASE", (b_phase, a_phase),
                    len(reached), branch_win, solid_exits(branch),
                    branch.levels_completed, flush=True,
                )


arena.run_program("dc22", observe)

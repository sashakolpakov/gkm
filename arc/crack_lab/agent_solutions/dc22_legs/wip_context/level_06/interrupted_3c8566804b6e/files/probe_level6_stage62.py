import importlib.util
import os
import sys
from collections import deque

import numpy as np

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

REMOTE = [
    3, 3, 3, 3, 3,
    A_CONTROL, A_CONTROL, A_CONTROL, A_CONTROL,
    2, 2, 3, 3, 3, 2, 3, A_CONTROL, 1, A_CONTROL, 1, 1,
    B_CONTROL,
] + [1] * 17 + [3] + [2] * 11


def step(env, action):
    env.step(*action) if isinstance(action, tuple) else env.step(action)


def apply(env, path):
    for action in path:
        step(env, action)


def avatar(env):
    rows, cols = np.where(env.frame()[:62, :40] == 14)
    if not len(rows):
        return None
    return int(rows.min() // 2), int(cols.min() // 2)


def left_world(env):
    frame = env.frame()[:62, :40].copy()
    frame[frame == 14] = 15
    return frame


def run(env):
    solver.solve(env)
    apply(env, REMOTE)
    queue = deque([(env.clone(), [])])
    nodes = {}
    while queue:
        node, path = queue.popleft()
        position = avatar(node)
        if position in nodes:
            continue
        nodes[position] = (node, path)
        for direction in (1, 2, 3, 4):
            child = node.clone()
            child.step(direction)
            if avatar(child) not in nodes:
                queue.append((child, path + [direction]))
    print("REACH", sorted(nodes), flush=True)

    outcomes = {}
    for position, (node, path) in sorted(nodes.items()):
        child = node.clone()
        before = left_world(child)
        step(child, B_CONTROL)
        after = left_world(child)
        delta = np.argwhere(before != after)
        signature = (
            int(len(delta)),
            None if not len(delta) else (
                int(delta[:, 0].min()), int(delta[:, 1].min()),
                int(delta[:, 0].max()), int(delta[:, 1].max()),
            ),
            avatar(child),
        )
        outcomes.setdefault(signature, []).append((position, path))
    for signature, entries in outcomes.items():
        print(
            "B_OUTCOME", signature,
            "positions", [position for position, _ in entries],
            "example", entries[0][1], flush=True,
        )


A.run_program("dc22", run)

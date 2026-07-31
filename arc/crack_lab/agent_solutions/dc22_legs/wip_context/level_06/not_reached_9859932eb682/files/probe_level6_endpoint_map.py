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
TOP = REMOTE + ROOT_TO_SELECTOR + [S, S, 3, B]
TOP_ISLAND = {
    (row, col)
    for row in range(2, 6)
    for col in range(2, 6)
}


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


def movement_destinations(root, limit=100):
    start_key = key(root)
    nodes = {start_key: (root.clone(), [])}
    queue = deque([start_key])
    destinations = {}
    win = None
    while queue and len(nodes) < limit:
        node_key = queue.popleft()
        node, path = nodes[node_key]
        position = avatar(node)
        if position not in TOP_ISLAND:
            destinations.setdefault(position, path)
        for direction in (1, 2, 3, 4):
            child = node.clone()
            child.step(direction)
            child_path = path + [direction]
            if child.levels_completed > 5:
                win = child_path
                return destinations, win, len(nodes)
            child_key = key(child)
            if child_key not in nodes:
                nodes[child_key] = (child, child_path)
                queue.append(child_key)
    return destinations, win, len(nodes)


def run(env):
    solver.solve(env)
    selector = env.clone()
    apply(selector, REMOTE + ROOT_TO_SELECTOR)
    print("SELECTOR", avatar(selector), selector.levels_completed, flush=True)
    for selector_phase in range(4):
        outbound = selector.clone()
        apply(outbound, [S] * selector_phase + [3, B])
        nodes, win, states = movement_destinations(outbound)
        print(
            "OUTBOUND", selector_phase, "avatar", avatar(outbound),
            "reachable_outside_top", sorted(nodes)[:12],
            "win", win, "states", states, flush=True,
        )

    apply(env, TOP)
    print("TOP", avatar(env), env.levels_completed, flush=True)
    for selector_phase in range(4):
        selected = env.clone()
        apply(selected, [S] * selector_phase)
        for assembly_phase in range(4):
            phased = selected.clone()
            apply(phased, [B] * assembly_phase)
            destinations, win, states = movement_destinations(phased)
            compact = sorted(
                (position, path)
                for position, path in destinations.items()
                if position is not None
            )
            print(
                "PHASE", (selector_phase, assembly_phase),
                "dest", compact[:12], "win", win, "states", states,
                flush=True,
            )


arena.run_program("dc22", run)

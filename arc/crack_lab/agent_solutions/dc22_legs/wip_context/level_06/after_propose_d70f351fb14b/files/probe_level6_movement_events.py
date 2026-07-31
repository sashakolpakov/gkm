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


def non_avatar_delta(before, after):
    changed = before != after
    old = before[changed]
    new = after[changed]
    mask = (old != 14) & (new != 14)
    if not np.any(mask):
        return 0, None
    rows, cols = np.where(changed)
    rows = rows[mask]
    cols = cols[mask]
    return (
        int(np.count_nonzero(mask)),
        (int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())),
    )


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


def scan(label, root, limit=220):
    start_key = key(root)
    nodes = {start_key: (root.clone(), [])}
    queue = deque([start_key])
    events = []
    positions = {avatar(root)}
    while queue and len(nodes) < limit:
        node_key = queue.popleft()
        node, path = nodes[node_key]
        for direction in (1, 2, 3, 4):
            child = node.clone()
            before = np.asarray(node.frame())[:62, :40]
            child.step(direction)
            child_path = path + [direction]
            positions.add(avatar(child))
            delta = non_avatar_delta(
                before, np.asarray(child.frame())[:62, :40]
            )
            goals = solid_goals(child)
            if delta[0] or goals or child.levels_completed > 5:
                events.append(
                    (
                        child_path, avatar(node), avatar(child),
                        delta, goals, child.levels_completed,
                    )
                )
            child_key = key(child)
            if child_key not in nodes:
                nodes[child_key] = (child, child_path)
                queue.append(child_key)
    print(
        label, "states", len(nodes), "positions", len(positions),
        "events", events, flush=True,
    )


def run(env):
    solver.solve(env)
    apply(env, REMOTE)
    scan("REMOTE", env)

    hub = env.clone()
    apply(hub, ROOT_TO_SELECTOR + [S, S, S, 3, B])
    scan("HUB", hub)

    top = env.clone()
    apply(top, ROOT_TO_SELECTOR + [S, S, 3, B])
    scan("TOP", top)

    network = hub.clone()
    apply(network, [B, 2, 2, 2, 2, A])
    scan("NETWORK", network)

    horizontal = network.clone()
    apply(horizontal, [1, 1, 1, 4, 4, B])
    scan("HORIZONTAL", horizontal)


arena.run_program("dc22", run)

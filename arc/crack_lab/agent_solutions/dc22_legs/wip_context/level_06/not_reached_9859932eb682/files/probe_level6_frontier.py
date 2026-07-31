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
REMOTE = (
    [3] * 5
    + [A] * 4
    + [2, 2, 3, 3, 3, 2, 3, A, 1, A, 1, 1, B]
    + [1] * 17
    + [3]
    + [2] * 11
)


def step(env, action):
    env.step(*action) if isinstance(action, tuple) else env.step(action)


def apply(env, actions):
    for action in actions:
        step(env, action)


def avatar(env):
    rows, cols = np.where(np.asarray(env.frame())[:62, :40] == 14)
    return None if not len(rows) else (int(rows.min() // 2), int(cols.min() // 2))


def non_avatar_change(before, after):
    changed = before != after
    if not np.any(changed):
        return 0
    old = before[changed]
    new = after[changed]
    return int(np.count_nonzero((old != 14) & (new != 14)))


def trace(root, label, actions):
    env = root.clone()
    events = []
    previous = avatar(env)
    for index, action in enumerate(actions, 1):
        before = np.asarray(env.frame())[:62, :40].copy()
        step(env, action)
        current = avatar(env)
        world = non_avatar_change(before, np.asarray(env.frame())[:62, :40])
        if current != previous or world or env.levels_completed > 5:
            events.append((index, action, previous, current, world, env.levels_completed))
        previous = current
        if env.levels_completed > 5:
            break
    print(label, "end", avatar(env), "events", events, flush=True)


def movement_nodes(root):
    start = avatar(root)
    nodes = {start: (root.clone(), [])}
    queue = deque([start])
    winning = None
    while queue:
        position = queue.popleft()
        node, path = nodes[position]
        for direction in (1, 2, 3, 4):
            child = node.clone()
            child.step(direction)
            child_position = avatar(child)
            if child.levels_completed > 5:
                winning = path + [direction]
                continue
            if child_position not in nodes:
                nodes[child_position] = (child, path + [direction])
                queue.append(child_position)
    return nodes, winning


def reach_metric(root):
    nodes, winning = movement_nodes(root)
    positions = [position for position in nodes if position is not None]
    return (
        len(positions),
        min(row for row, _ in positions),
        max(row for row, _ in positions),
        min(col for _, col in positions),
        max(col for _, col in positions),
        winning,
    )


def run(env):
    solver.solve(env)
    apply(env, REMOTE)
    apply(env, [1, 1, 1, 4, 4])
    print("ROOT", avatar(env), env.levels_completed, flush=True)

    for phase in range(7):
        trace(env, f"PHASE_{phase}", [B] * phase + [4] * 12)
    trace(env, "SYNC_BR", [B, 4] * 12)
    trace(env, "SYNC_RB", [4, B] * 12)

    crossed = env.clone()
    apply(crossed, [B, 4, 4, 4])
    nodes, winning = movement_nodes(crossed)
    print("CROSSED_REACH", sorted(nodes), "win", winning, flush=True)
    for label, control in (("A", A), ("B", B)):
        groups = {}
        for position, (node, path) in nodes.items():
            before = np.asarray(node.frame())[:62, :40].copy()
            child = node.clone()
            step(child, control)
            after = np.asarray(child.frame())[:62, :40]
            delta = non_avatar_change(before, after)
            metric = reach_metric(child)
            signature = (delta, metric)
            groups.setdefault(signature, []).append((position, path))
        for signature, entries in groups.items():
            print(
                "HANDOFF", label,
                "effect", signature,
                "positions", [entry[0] for entry in entries],
                "example", entries[0],
                flush=True,
            )


arena.run_program("dc22", run)

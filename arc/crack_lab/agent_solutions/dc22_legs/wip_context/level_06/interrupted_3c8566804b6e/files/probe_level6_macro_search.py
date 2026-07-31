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
D_BY_POSITION = {
    (28, 17): (6, 50, 32),
    (30, 17): (6, 50, 40),
    (29, 16): (6, 46, 36),
    (29, 18): (6, 54, 36),
}
B_SPECIAL_POSITIONS = {
    (2, 2),
    (9, 3), (9, 4),
    (16, 5), (17, 5),
    (16, 8), (17, 8),
    (24, 9),
    (26, 16),
    (29, 17),
}
A_SPECIAL_POSITIONS = {
    (row, col)
    for row in range(26, 30)
    for col in range(3, 11)
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


def step(env, action):
    env.step(*action) if isinstance(action, tuple) else env.step(action)


def apply(env, actions):
    for action in actions:
        step(env, action)


def world_key(env):
    return np.asarray(env.frame())[:62, :40].tobytes()


def avatar(env):
    rows, cols = np.where(np.asarray(env.frame())[:62, :40] == 14)
    return None if not len(rows) else (int(rows.min() // 2), int(cols.min() // 2))


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


def movement_closure(root, max_states=180):
    start = root.clone()
    start_key = world_key(start)
    nodes = {start_key: (start, [])}
    queue = deque([start_key])
    while queue and len(nodes) < max_states:
        key = queue.popleft()
        node, path = nodes[key]
        for direction in (1, 2, 3, 4):
            child = node.clone()
            child.step(direction)
            child_path = path + [direction]
            if child.levels_completed > 5:
                return nodes, child_path
            child_key = world_key(child)
            if child_key not in nodes:
                nodes[child_key] = (child, child_path)
                queue.append(child_key)
    return nodes, None


def first_changing(nodes, control):
    ordered = sorted(
        nodes.values(),
        key=lambda item: (len(item[1]), avatar(item[0]) or (99, 99)),
    )
    for node, walk in ordered:
        child = node.clone()
        before = world_key(child)
        step(child, control)
        if child.levels_completed > 5 or world_key(child) != before:
            return child, walk
    return None


def at_position(nodes, position):
    choices = [
        (node, path)
        for node, path in nodes.values()
        if avatar(node) == position
    ]
    return min(choices, key=lambda item: len(item[1])) if choices else None


def run(env):
    solver.solve(env)
    apply(env, HUB)
    base_level = env.levels_completed
    queue = deque([(env.clone(), [])])
    seen = set()
    queued = {world_key(env)}
    best = (99, -1, 0)

    while queue and len(seen) < 600:
        root, root_path = queue.popleft()
        nodes, winning_walk = movement_closure(root)
        if winning_walk is not None:
            print("WIN", root_path + winning_walk, "states", len(seen), flush=True)
            return
        goals = sorted({
            goal
            for node, _ in nodes.values()
            for goal in solid_goals(node)
        })
        if goals:
            print(
                "GOAL_REVEALED", goals, "path", root_path,
                "states", len(seen), flush=True,
            )
            return
        canonical_key = min(nodes)
        if canonical_key in seen:
            continue
        seen.add(canonical_key)

        positions = {
            avatar(node)
            for node, _ in nodes.values()
            if avatar(node) is not None
        }
        metric = (
            min(row for row, _ in positions),
            max(col for _, col in positions),
            len(positions),
        )
        if (
            metric[0] < best[0]
            or metric[1] > best[1]
            or metric[2] > best[2]
        ):
            best = (
                min(best[0], metric[0]),
                max(best[1], metric[1]),
                max(best[2], metric[2]),
            )
            print(
                "PROGRESS", metric, "best", best,
                "path_len", len(root_path), "path", root_path,
                flush=True,
            )

        candidates = []
        for control in (A, B, S):
            changed = first_changing(nodes, control)
            if changed is not None:
                child, walk = changed
                candidates.append((child, root_path + walk + [control]))

        for position in B_SPECIAL_POSITIONS:
            choice = at_position(nodes, position)
            if choice is None:
                continue
            node, walk = choice
            child = node.clone()
            before = world_key(child)
            step(child, B)
            if child.levels_completed > base_level or world_key(child) != before:
                candidates.append((child, root_path + walk + [B]))

        for position in A_SPECIAL_POSITIONS:
            choice = at_position(nodes, position)
            if choice is None:
                continue
            node, walk = choice
            child = node.clone()
            before = world_key(child)
            step(child, A)
            if child.levels_completed > base_level or world_key(child) != before:
                candidates.append((child, root_path + walk + [A]))

        for position, control in D_BY_POSITION.items():
            choice = at_position(nodes, position)
            if choice is None:
                continue
            node, walk = choice
            child = node.clone()
            before = world_key(child)
            step(child, control)
            if child.levels_completed > base_level or world_key(child) != before:
                candidates.append((child, root_path + walk + [control]))

        resulting = set()
        for child, path in candidates:
            if child.levels_completed > base_level:
                print("WIN", path, "states", len(seen), flush=True)
                return
            if len(path) > 180:
                continue
            key = world_key(child)
            if key in resulting or key in queued:
                continue
            resulting.add(key)
            queued.add(key)
            queue.append((child, path))

        if len(seen) % 25 == 0:
            print(
                "STATES", len(seen), "queue", len(queue),
                "best", best, flush=True,
            )
    print(
        "DONE", len(seen), "queue", len(queue), "best", best,
        flush=True,
    )


arena.run_program("dc22", run)

"""BFS ring routes without collapsing persistent dock/reveal state."""
from collections import deque
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

import perception
import solve
from probe_l6_right import avatar_position, enter_right


MOVES = {
    "U": (1, (6, 50, 34), 2),
    "D": (2, (6, 50, 40), 1),
    "L": (3, (6, 46, 36), 4),
    "R": (4, (6, 54, 36), 3),
}


def key(env):
    return perception.arr(env.frame())[:63].tobytes()


def goal_signature(env):
    frame = perception.arr(env.frame())
    return (
        int((frame[:63, :40] == 11).sum()),
        tuple(
            (blob.bbox, blob.area)
            for blob in perception.connected_components(
                frame[:, :40], colors=(11,), min_area=1
            )
        ),
    )


def ring_signature(env):
    return tuple(
        (blob.bbox, blob.area)
        for blob in perception.connected_components(
            env.frame(), colors=(8,), min_area=20
        )
        if blob.bbox[1] < 32
    )


def observe(env):
    solve.solve(env)
    root = enter_right(env, 3)
    base_level = root.levels_completed
    initial_goal = goal_signature(root)
    queue = deque([(root.clone(), [])])
    seen = {key(root)}
    print(
        "RING_HISTORY_START", avatar_position(root),
        ring_signature(root), initial_goal,
    )
    while queue and len(seen) < 800:
        node, path = queue.popleft()
        if len(path) >= 18:
            continue
        for name, (outward, control, inward) in MOVES.items():
            child = node.clone()
            child.step(outward)
            child.step(*control)
            child.step(inward)
            child_path = path + [name]
            if child.levels_completed > base_level:
                print("RING_HISTORY_WIN", child_path)
                return
            child_key = key(child)
            if child_key in seen:
                continue
            seen.add(child_key)
            signature = goal_signature(child)
            if signature != initial_goal:
                print(
                    "RING_HISTORY_GOAL_CHANGE", child_path,
                    ring_signature(child), signature,
                )
            if len(seen) % 50 == 0:
                print(
                    "RING_HISTORY_PROGRESS", len(seen), len(child_path),
                    ring_signature(child), signature[0],
                )
            queue.append((child, child_path))
    print("RING_HISTORY_DONE", len(seen), len(queue))


arena.run_program("dc22", observe)

"""Bounded macro-BFS over remote cargo positions and main orientation."""
from collections import deque
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

import perception
import solve
from probe_l6_right import MAIN, avatar_position, enter_right


CENTER = (58, 34)
DIRECTIONS = {
    "U": (1, (6, 50, 34)),
    "D": (2, (6, 50, 40)),
    "L": (3, (6, 46, 36)),
    "R": (4, (6, 54, 36)),
}
TO_CENTER = {
    (56, 34): 2,
    (60, 34): 1,
    (58, 32): 4,
    (58, 36): 3,
}


def key(env):
    frame = perception.arr(env.frame())
    return (
        frame[8:40, :40].tobytes(),
        avatar_position(env),
        int(frame[4, 4]),
    )


def cargo_components(env):
    return [
        (blob.bbox, blob.area)
        for blob in perception.connected_components(
            env.frame(), colors=(8,), min_area=20
        )
        if blob.bbox[1] < 32
    ]


def observe(env):
    solve.solve(env)
    root = enter_right(env, 3)
    base_level = root.levels_completed
    queue = deque([(root.clone(), [])])
    seen = {key(root)}
    placements = {tuple(cargo_components(root))}
    print("CARGO_BFS_START", cargo_components(root), avatar_position(root))
    while queue and len(seen) < 350:
        node, path = queue.popleft()
        position = avatar_position(node)
        for name, (movement, control) in DIRECTIONS.items():
            child = node.clone()
            child_path = list(path)
            if position != CENTER:
                back = TO_CENTER.get(position)
                if back is None:
                    continue
                child.step(back)
                child_path.append(back)
            child.step(movement)
            child.step(*control)
            child_path.extend((movement, control))
            if child.levels_completed > base_level:
                print("CARGO_BFS_WIN", child_path)
                return
            child_key = key(child)
            if child_key in seen:
                continue
            seen.add(child_key)
            placement = tuple(cargo_components(child))
            if placement not in placements:
                placements.add(placement)
                print("CARGO_PLACEMENT", placement, child_path)
            queue.append((child, child_path))
        child = node.clone()
        child.step(*MAIN)
        child_path = path + [MAIN]
        if child.levels_completed > base_level:
            print("CARGO_BFS_WIN", child_path)
            return
        child_key = key(child)
        if child_key not in seen:
            seen.add(child_key)
            queue.append((child, child_path))
    print("CARGO_BFS_DONE", len(seen), len(placements))


arena.run_program("dc22", observe)

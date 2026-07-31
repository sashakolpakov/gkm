"""Bounded observational BFS over level 6's discovered affordances."""
from collections import deque
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

import perception
import solve


ACTIONS = (1, 2, 3, 4, (6, 54, 6), (6, 50, 26))


def avatar_position(env):
    blobs = perception.connected_components(env.frame(), colors=(14,), min_area=1)
    avatars = [
        blob for blob in blobs
        if blob.area >= 2 and blob.bbox[1] < 32
    ]
    return avatars[0].top_left if avatars else None


def key(env):
    return perception.arr(env.frame())[:63].tobytes()


def observe(env):
    solve.solve(env)
    base_level = env.levels_completed
    queue = deque([(env.clone(), [])])
    seen = {key(env)}
    positions = {avatar_position(env)}
    best_row = avatar_position(env)[0]
    print("BFS_START", base_level, avatar_position(env))
    while queue and len(seen) < 6000:
        node, path = queue.popleft()
        if len(path) >= 100 or node.terminal():
            continue
        for action in ACTIONS:
            child = node.clone()
            if isinstance(action, tuple):
                child.step(*action)
            else:
                child.step(action)
            child_path = path + [action]
            if child.levels_completed > base_level:
                print("FOUND", len(child_path), child_path)
                return
            child_key = key(child)
            if child_key in seen:
                continue
            seen.add(child_key)
            position = avatar_position(child)
            positions.add(position)
            if position is not None and position[0] < best_row:
                best_row = position[0]
                print("PROGRESS", len(child_path), position, child_path)
            if len(seen) % 500 == 0:
                print("STATES", len(seen), "POSITIONS", len(positions), "DEPTH", len(child_path))
            queue.append((child, child_path))
    print("NO_PATH", len(seen), len(positions), best_row)


arena.run_program("dc22", observe)

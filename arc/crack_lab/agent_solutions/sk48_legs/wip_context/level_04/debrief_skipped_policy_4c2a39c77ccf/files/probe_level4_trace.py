"""Compact symbolic trace and bounded replay search for sk48 level 4."""
import sys
from heapq import heappop, heappush

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A

from perception import arr, connected_components
from players import play_level_1, play_level_2, play_level_3


def advance(env):
    play_level_1(env)
    play_level_2(env)
    play_level_3(env)


def pieces(env):
    found = []
    for blob in connected_components(
        env.frame(), colors=(0, 6, 8, 9, 12, 14), min_area=4
    ):
        if blob.bbox[0] < 53:
            found.append((blob.color, blob.bbox))
    return tuple(found)


def tokens(env):
    return tuple(item for item in pieces(env) if item[0] in (8, 9, 12, 14))


def compact(env):
    return (
        env.levels_completed,
        tuple((color, box[0], box[1]) for color, box in pieces(env)),
    )


def train_distance(node):
    positions = {
        color: (box[0], box[1])
        for color, box in tokens(node)
    }
    if set(positions) != {8, 9, 12, 14}:
        return 999
    return min(
        abs(positions[8][0] - left_row) // 6
        + abs(positions[8][1] - 12) // 6
        + abs(positions[12][0] - left_row) // 6
        + abs(positions[12][1] - 18) // 6
        + abs(positions[9][0] - right_row) // 6
        + abs(positions[9][1] - 42) // 6
        + abs(positions[14][0] - right_row) // 6
        + abs(positions[14][1] - 48) // 6
        for left_row in (7, 13, 19)
        for right_row in (7, 13, 19)
    )


def search(env, prefix, max_states=6000, max_depth=36):
    root = env.clone()
    for action in prefix:
        root.step(action)

    def rebuild(path):
        node = root.clone()
        for action in path:
            node.step(action)
        return node

    serial = 0
    queue = [(train_distance(root), 0, serial, ())]
    seen = {pieces(root)}
    best = train_distance(root)
    while queue and len(seen) < max_states:
        _, _, _, path = heappop(queue)
        node = rebuild(path)
        if node.levels_completed > 3:
            return path, len(seen)
        if len(path) >= max_depth:
            continue
        for action in (1, 2, 3, 4):
            child = node.clone()
            child.step(action)
            key = pieces(child)
            if key in seen:
                continue
            seen.add(key)
            child_path = path + (action,)
            if child.levels_completed > 3:
                return child_path, len(seen)
            distance = train_distance(child)
            if distance < best:
                best = distance
                print("PROGRESS", len(seen), distance, child_path, compact(child))
            serial += 1
            heappush(
                queue,
                (distance + len(child_path) // 4, len(child_path), serial, child_path),
            )
    return None, len(seen)


def inspect(env):
    advance(env)
    print("START", compact(env))

    trace = (1, 1, 1, 4, 1, 3, 1)
    node = env.clone()
    for action in trace:
        node.step(action)
        print("TRACE", action, compact(node))

    path, states = search(env, (), max_states=8000, max_depth=44)
    print("SEARCH", states, path)
    if path:
        for action in path:
            node.step(action)
        print("RESULT", compact(node))


if __name__ == "__main__":
    levels, path, err = A.run_program("sk48", inspect)
    print("END", levels, len(path), err)

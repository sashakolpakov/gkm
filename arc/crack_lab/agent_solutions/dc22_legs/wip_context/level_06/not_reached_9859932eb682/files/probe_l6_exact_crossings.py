"""Exact-frame traversal from the horizontal rotator at every ring placement."""
from collections import deque
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

import perception
import solve
from probe_l6_ring_route import HUB_TO_BRIDGE
from probe_l6_right import MAIN, TOP, avatar_position, enter_right


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


def placement_key(env):
    return perception.arr(env.frame())[6:42, 14:34].tobytes()


def placement_label(env):
    return [
        (blob.bbox, blob.area)
        for blob in perception.connected_components(
            env.frame(), colors=(8,), min_area=20
        )
        if blob.bbox[1] < 32
    ]


def placements_with_paths(root):
    queue = deque([(root.clone(), [])])
    seen = {placement_key(root)}
    out = [(root.clone(), [])]
    while queue:
        node, path = queue.popleft()
        position = avatar_position(node)
        for movement, control in DIRECTIONS.values():
            child = node.clone()
            child_path = list(path)
            if position != CENTER:
                child.step(TO_CENTER[position])
                child_path.append(TO_CENTER[position])
            child.step(movement)
            child.step(*control)
            child_path.extend((movement, control))
            key = placement_key(child)
            if key in seen:
                continue
            seen.add(key)
            out.append((child.clone(), child_path))
            queue.append((child, child_path))
    return out


def horizontal_entry(placement):
    node = placement.clone()
    position = avatar_position(node)
    if position != CENTER:
        node.step(TO_CENTER[position])
    node.step(*MAIN)
    for action in HUB_TO_BRIDGE:
        node.step(action)
    node.step(*TOP)
    node.step(1)
    node.step(3)
    node.step(*TOP)
    for _ in range(12):
        node.step(1)
    node.step(*MAIN)
    return node


def exact_reach(root, max_states=300, max_depth=40):
    base_level = root.levels_completed
    queue = deque([(root.clone(), [])])
    seen = {perception.arr(root.frame())[:63].tobytes()}
    partial = set()
    while queue and len(seen) < max_states:
        node, path = queue.popleft()
        if len(path) >= max_depth:
            continue
        for action in (1, 2, 3, 4, MAIN):
            child = node.clone()
            if isinstance(action, tuple):
                child.step(*action)
            else:
                child.step(action)
            child_path = path + [action]
            if child.levels_completed > base_level:
                return child_path, len(seen), partial
            frame = perception.arr(child.frame())
            blobs = perception.connected_components(
                frame, colors=(14,), min_area=1
            )
            partial.update(
                (blob.bbox, blob.area)
                for blob in blobs
                if blob.bbox[1] < 40 and blob.area < 4
            )
            key = frame[:63].tobytes()
            if key in seen:
                continue
            seen.add(key)
            queue.append((child, child_path))
    return None, len(seen), partial


def observe(env):
    solve.solve(env)
    placements = placements_with_paths(enter_right(env, 3))
    print("EXACT_PLACEMENTS", len(placements))
    for index, (placement, cargo_path) in enumerate(placements):
        if index not in (9, 11, 13):
            continue
        root = horizontal_entry(placement)
        win, states, partial = exact_reach(root)
        print(
            "EXACT_CROSSING", index, placement_label(placement),
            "STATES", states, "PARTIAL", len(partial), "WIN", win,
        )
        if win is not None:
            print("EXACT_CARGO_PATH", cargo_path)
            return


if __name__ == "__main__":
    arena.run_program("dc22", observe)

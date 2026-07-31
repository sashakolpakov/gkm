"""Exact bounded movement/main BFS from the staged ring overlap."""
from collections import deque
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

import perception
import solve
from probe_l6_bridge_avatar import HUB_TO_BRIDGE
from probe_l6_right import MAIN, TOP, enter_right


LEFT = (6, 46, 36)


def enter_overlap(env):
    node = enter_right(env, 3)
    node.step(3)
    node.step(*LEFT)
    node.step(4)
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
    for _ in range(5):
        node.step(4)
    return node


def avatar_pixels(env):
    frame = perception.arr(env.frame())
    ys, xs = (frame[:63, :40] == 14).nonzero()
    return tuple((int(y), int(x)) for y, x in zip(ys, xs))


def observe(env):
    solve.solve(env)
    root = enter_overlap(env)
    base_level = root.levels_completed
    queue = deque([(root.clone(), [])])
    seen = {perception.arr(root.frame())[:63].tobytes()}
    signatures = {avatar_pixels(root)}
    while queue and len(seen) < 500:
        node, path = queue.popleft()
        if len(path) >= 30:
            continue
        for action in (1, 2, 3, 4, MAIN):
            child = node.clone()
            if isinstance(action, tuple):
                child.step(*action)
            else:
                child.step(action)
            child_path = path + [action]
            if child.levels_completed > base_level:
                print("ABOARD_BFS_WIN", child_path)
                return
            key = perception.arr(child.frame())[:63].tobytes()
            if key in seen:
                continue
            seen.add(key)
            signature = avatar_pixels(child)
            if signature not in signatures:
                signatures.add(signature)
                print("ABOARD_POSITION", signature, child_path)
            queue.append((child, child_path))
    print("ABOARD_BFS_DONE", len(seen), len(signatures))


if __name__ == "__main__":
    arena.run_program("dc22", observe)

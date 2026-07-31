"""Verify the complete matched-pad route on a pristine level-6 clone."""
from collections import deque
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

import perception
import solve


TOP = (6, 54, 6)
MAIN = (6, 50, 26)
SELECTOR = (6, 50, 46)
TO_BRIDGE = [3, 3, 3, 3, 3, 2, 2, 3, 3, 3, 2, 3]
TO_REMOTE_PAD = [4, 1, 4, 1, 4, 4, 1, 1, 1]


def avatar_position(env):
    for blob in perception.connected_components(
            env.frame(), colors=(14,), min_area=2):
        if blob.bbox[1] < 32:
            return blob.top_left
    return None


def pad(env):
    return tuple(
        int(value)
        for value in perception.arr(env.frame())[48:50, 18:20].flat
    )


def observe(env):
    solve.solve(env)
    node = env.clone()
    base_level = node.levels_completed
    for _ in range(4):
        node.step(*TOP)
    for action in TO_BRIDGE:
        node.step(action)
    node.step(*TOP)
    node.step(1)
    node.step(3)
    node.step(*TOP)
    for _ in range(6):
        node.step(1)
    node.step(*MAIN)
    for _ in range(13):
        node.step(1)
    print("GLYPH", avatar_position(node), pad(node), node.levels_completed)
    node.step(*SELECTOR)
    node.step(*SELECTOR)
    print("MATCHED", avatar_position(node), pad(node), node.levels_completed)
    for _ in range(13):
        node.step(2)
    for _ in range(6):
        node.step(2)
    node.step(*TOP)
    node.step(*TOP)
    node.step(4)
    node.step(*TOP)
    node.step(2)
    node.step(*TOP)
    for action in TO_REMOTE_PAD:
        node.step(action)
    print(
        "REMOTE_ENTER", avatar_position(node), node.levels_completed,
        "GAIN", node.levels_completed - base_level, "PAD", pad(node),
    )
    for click in range(1, 5):
        node.step(*SELECTOR)
        print(
            "REMOTE_CLICK", click, avatar_position(node),
            node.levels_completed, node.levels_completed - base_level,
        )
        if node.levels_completed > base_level:
            break
    node.step(*MAIN)
    print(
        "REMOTE_MAIN", avatar_position(node), node.levels_completed,
        node.levels_completed - base_level,
    )
    for selector_steps in range(4):
        branch = node.clone()
        for _ in range(selector_steps):
            branch.step(*SELECTOR)
        branch.step(*MAIN)
        print(
            "TOP_MAIN_BRANCH", selector_steps, avatar_position(branch),
            branch.levels_completed, pad(branch),
        )
    top_frame = perception.arr(node.frame()).copy()
    for y in range(2, 64, 4):
        for x in range(2, 64, 4):
            probe = node.clone()
            probe.step(6, x, y)
            delta = perception.frame_delta(top_frame, probe.frame())
            samples = [
                sample for sample in delta["samples"] if sample[0] < 63
            ]
            if samples:
                print("TOP_CLICK", (x, y), delta["count"], delta["bbox"], samples[:8])
    queue = deque([(node.clone(), [])])
    seen = {avatar_position(node)}
    while queue:
        current, path = queue.popleft()
        for action in (1, 2, 3, 4):
            child = current.clone()
            child.step(action)
            child_path = path + [action]
            if child.levels_completed > base_level:
                print("FINAL_PATH", child_path, child.levels_completed)
                return
            position = avatar_position(child)
            if position not in seen:
                seen.add(position)
                queue.append((child, child_path))
    print("FINAL_NO_REWARD", sorted(seen))


arena.run_program("dc22", observe)

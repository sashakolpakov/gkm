"""Bounded four-turn search from the verified staged lane-six frontier."""

import sys
from collections import deque

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import connected_components
from probe_l9_col5_depth6_actions import col5_depth6
from probe_l9_control_row import compact
from probe_l9_route_deletions import enter_level_9


def step(env, action):
    env.step(*action) if isinstance(action, tuple) else env.step(action)


def frontier(root):
    child = col5_depth6(root)
    for action in ((6, 39, 33), (6, 39, 27), 4):
        step(child, action)
    return child


def key(env, depth):
    frame = np.asarray(env.frame()).copy()
    frame[63, :] = 0
    return depth, int(env.levels_completed), bool(env.terminal()), frame.tobytes()


def actions(env):
    result = [3, 4]
    frame = env.frame()
    for blob in connected_components(
        frame, colors=(7, 8, 12, 14, 15), min_area=3
    ):
        if blob.bbox[0] >= 63 or blob.area != 21:
            continue
        result.append(
            (6, round(blob.centroid[1]), round(blob.centroid[0]))
        )
    avatars = connected_components(frame, colors=(9,), min_area=3)
    if avatars:
        row = round((avatars[0].centroid[0] - 3) / 6)
        col = round((avatars[0].centroid[1] - 3) / 6)
        for r in range(max(0, row - 2), min(10, row + 3)):
            for c in range(max(0, col - 2), min(10, col + 3)):
                result.append((6, 3 + 6 * c, 3 + 6 * r))
    return tuple(dict.fromkeys(result))


def score(env):
    visible_goal = any(
        blob.bbox[0] < 63
        for blob in connected_components(env.frame(), colors=(7,), min_area=2)
    )
    yellow = [
        blob.bbox[0]
        for blob in connected_components(env.frame(), colors=(14,), min_area=3)
        if blob.bbox[0] < 63
    ]
    return (
        int(env.levels_completed) >= 9,
        visible_goal,
        not env.terminal(),
        60 - min(yellow) if yellow else 60,
    )


def probe(env):
    enter_level_9(env)
    root = frontier(env)
    print("ROOT", compact(root), "actions", len(actions(root)))
    queue = deque([(root, ())])
    seen = {key(root, 0)}
    endpoints = []
    while queue and len(seen) <= 10000:
        node, path = queue.popleft()
        if int(node.levels_completed) >= 9:
            print("WIN", path, compact(node), "states", len(seen))
            return
        if len(path) == 4 or node.terminal():
            endpoints.append((score(node), path, compact(node)))
            continue
        for action in actions(node):
            child = node.clone()
            step(child, action)
            child_path = path + (action,)
            if int(child.levels_completed) >= 9:
                print("WIN", child_path, compact(child), "states", len(seen))
                return
            child_key = key(child, len(child_path))
            if child_key in seen:
                continue
            seen.add(child_key)
            queue.append((child, child_path))
    print("STATES", len(seen), "QUEUE", len(queue))
    for item in sorted(endpoints, reverse=True)[:12]:
        print("BEST", item)


if __name__ == "__main__":
    arena.run_program("bp35", probe)

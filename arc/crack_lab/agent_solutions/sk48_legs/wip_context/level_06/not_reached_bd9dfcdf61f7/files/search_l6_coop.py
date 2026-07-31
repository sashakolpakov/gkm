import heapq
import itertools
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

import players
from perception import connected_components


FIRST_EIGHT = (
    [2] + [4] * 6 + [3] * 4 + [1] * 2 + [4] + [1] * 2 + [3] * 3
    + [(6, 32, 5), 2, 1]
)
EIGHT_TARGETS = ((10, 32), (16, 32), (22, 32))


def reach_level_6(env):
    for level in range(1, 6):
        getattr(players, f"play_level_{level}")(env)
        assert env.levels_completed == level


def live_positions(env, color):
    return tuple(sorted(
        (round(blob.centroid[0]), round(blob.centroid[1]))
        for blob in connected_components(env.frame(), colors=(color,), min_area=12)
        if blob.centroid[0] < 53
    ))


def assignment_distance(points, targets):
    return min(
        sum(abs(r - tr) + abs(c - tc)
            for (r, c), (tr, tc) in zip(points, ordering))
        for ordering in itertools.permutations(targets)
    ) // 6


def collector_click(env, mode):
    color = 6 if mode == "h" else 15
    heads = [
        blob for blob in connected_components(env.frame(), colors=(color,), min_area=16)
        if blob.centroid[0] < 53
    ]
    head = min(heads, key=lambda blob: blob.centroid[1] if mode == "h"
               else blob.centroid[0])
    return (6, round(head.centroid[1]), round(head.centroid[0]))


def search(root, max_states=8000, max_depth=75):
    start = root.clone()
    for action in FIRST_EIGHT:
        start.step(*action) if isinstance(action, tuple) else start.step(action)
    start_mode = "v"
    serial = itertools.count()
    start_points = live_positions(start, 8)
    heap = [(4 * assignment_distance(start_points, EIGHT_TARGETS),
             0, next(serial), start, start_mode, ())]
    seen = {(np.asarray(start.frame()).tobytes(), start_mode): 0}
    expanded = 0
    while heap and expanded < max_states:
        _, depth, _, node, mode, path = heapq.heappop(heap)
        expanded += 1
        points = live_positions(node, 8)
        if set(points) == set(EIGHT_TARGETS):
            return list(path), expanded
        if depth >= max_depth:
            continue
        options = [1, 2, 3, 4, "switch"]
        for option in options:
            child = node.clone()
            child_mode = mode
            if option == "switch":
                child_mode = "h" if mode == "v" else "v"
                action = collector_click(child, child_mode)
                child.step(*action)
            else:
                action = option
                child.step(action)
            key = (np.asarray(child.frame()).tobytes(), child_mode)
            old_depth = seen.get(key)
            if old_depth is not None and old_depth <= depth + 1:
                continue
            seen[key] = depth + 1
            child_path = path + (action,)
            distance = assignment_distance(live_positions(child, 8), EIGHT_TARGETS)
            score = depth + 1 + 4 * distance
            heapq.heappush(
                heap,
                (score, depth + 1, next(serial), child, child_mode, child_path),
            )
    return None, expanded


def probe(env):
    reach_level_6(env)
    path, expanded = search(env)
    print("FOUND", path, "EXPANDED", expanded)


A.run_program("sk48", probe)

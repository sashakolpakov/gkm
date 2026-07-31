import heapq
import itertools
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

import players
from perception import connected_components


E1 = [2] + [4] * 6 + [3] * 4 + [1] * 2 + [4] + [1] * 2 + [3] * 3
EN = [2] * 5 + [4] * 6 + [3] * 4 + [1] * 4 + [4]
N1 = [4] + [2] * 6 + [1] * 4 + [3] * 2 + [2] + [3] * 2 + [1] * 3
NN = [4] * 5 + [2] * 6 + [1] * 4 + [3] * 4 + [2]
TARGETS = {
    8: ((10, 32), (16, 32), (22, 32)),
    9: ((28, 14), (28, 20), (28, 26)),
}


def reach_level_6(env):
    for level in range(1, 6):
        getattr(players, f"play_level_{level}")(env)
        assert env.levels_completed == level


def click(env, mode):
    color = 6 if mode == "h" else 15
    heads = [
        blob
        for blob in connected_components(env.frame(), colors=(color,), min_area=16)
        if blob.centroid[0] < 53
    ]
    head = min(
        heads,
        key=lambda blob: blob.centroid[1] if mode == "h" else blob.centroid[0],
    )
    return (6, round(head.centroid[1]), round(head.centroid[0]))


def apply(env, path):
    for action in path:
        if action in ("h", "v"):
            env.step(*click(env, action))
        else:
            env.step(action)


def positions(env, color):
    return tuple(
        sorted(
            (round(blob.centroid[0]), round(blob.centroid[1]))
            for blob in connected_components(env.frame(), colors=(color,), min_area=12)
            if blob.centroid[0] < 53
        )
    )


def heuristic(env):
    total_distance = 0
    exact = 0
    for color in (8, 9):
        points = positions(env, color)
        targets = TARGETS[color]
        if len(points) != 3:
            return 1000
        exact += len(set(points) & set(targets))
        total_distance += min(
            sum(
                abs(row - target_row) + abs(col - target_col)
                for (row, col), (target_row, target_col) in zip(points, ordering)
            )
            for ordering in itertools.permutations(targets)
        ) // 6
    return 3 * total_distance + 4 * (6 - exact)


def finish_root(env):
    chunks = (
        ["h"] + E1,
        ["h"] + EN + ["v", 2, "h", 3, 3, 3],
        ["h"] + EN + ["v", 2, "h", 3, 3, 3],
        ["v"] + N1,
        ["v"] + NN + ["h", 4, "v", 1, 1, 1],
        ["v"] + NN + ["h", 4, "v", 1, 1, 1],
    )
    root = env.clone()
    for chunk in chunks:
        apply(root, chunk)
    return root


def search(root, max_states=12000, max_depth=80):
    serial = itertools.count()
    start_mode = "v"
    heap = [(heuristic(root), 0, next(serial), root, start_mode, ())]
    seen = {(np.asarray(root.frame()).tobytes(), start_mode): 0}
    best = (heuristic(root), (), positions(root, 8), positions(root, 9))
    expanded = 0
    while heap and expanded < max_states:
        _, depth, _, node, mode, path = heapq.heappop(heap)
        expanded += 1
        value = heuristic(node)
        if value < best[0]:
            best = (value, path, positions(node, 8), positions(node, 9))
        if node.levels_completed > 5:
            return path, expanded, best
        if depth >= max_depth:
            continue
        for option in (1, 2, 3, 4, "switch"):
            child = node.clone()
            child_mode = mode
            if option == "switch":
                child_mode = "h" if mode == "v" else "v"
                action = click(child, child_mode)
                child.step(*action)
            else:
                action = option
                child.step(action)
            key = (np.asarray(child.frame()).tobytes(), child_mode)
            old = seen.get(key)
            if old is not None and old <= depth + 1:
                continue
            seen[key] = depth + 1
            child_path = path + (action,)
            priority = depth + 1 + heuristic(child)
            heapq.heappush(
                heap,
                (priority, depth + 1, next(serial), child, child_mode, child_path),
            )
    return None, expanded, best


def probe(env):
    reach_level_6(env)
    path, expanded, best = search(finish_root(env))
    print("FOUND", path, "EXPANDED", expanded)
    print("BEST", best)


A.run_program("sk48", probe)

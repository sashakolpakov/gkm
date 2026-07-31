import heapq
import itertools
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

import players
from perception import arr, connected_components


SMALL_TARGETS = ((30, 51), (54, 39))
LARGE_TARGET = (12, 39)


def centers(env, color):
    if color == 14:
        frame = arr(env.frame())
        inactive = list(zip(*np.where(frame[:63] == 5)))
        zeros = list(zip(*np.where(frame[:63] == 0)))
        def rim_score(p):
            r, c = p
            return int(np.count_nonzero(frame[max(0, r-1):r+2, max(0, c-1):c+2] == 14))
        active = max(zeros, key=rim_score)
        return [(int(active[0]), int(active[1]))] + [
            (int(r), int(c)) for r, c in inactive
        ]
    return [
        (round(b.centroid[0]), round(b.centroid[1]))
        for b in connected_components(env.frame(), colors=(color,), min_area=4)
    ]


def metric(env):
    small = centers(env, 14)
    large_all = centers(env, 11)
    if len(small) != 2 or not large_all:
        return 999
    large = large_all[0]
    assignments = (
        sum(abs(small[i][0] - SMALL_TARGETS[p[i]][0]) +
            abs(small[i][1] - SMALL_TARGETS[p[i]][1]) for i in range(2))
        for p in ((0, 1), (1, 0))
    )
    return min(assignments) + abs(large[0] - LARGE_TARGET[0]) + abs(large[1] - LARGE_TARGET[1])


def active_center(env):
    return centers(env, 14)[0]


def key(env):
    return arr(env.frame())[:63].tobytes()


def probe(env):
    players.play_level_1(env)
    players.play_level_2(env)
    players.play_level_3(env)
    root = env.clone()
    serial = itertools.count()
    heap = [(metric(root), 0, next(serial), root, [])]
    seen = {key(root): 0}
    best = metric(root)
    print("start_metric", best)
    while heap and len(seen) < 20000:
        _, cost, _, node, path = heapq.heappop(heap)
        if node.levels_completed > 3:
            print("FOUND", len(path), path)
            return
        for action in (1, 2, 3, 4):
            child = node.clone()
            child.step(action)
            child_path = path + [action]
            child_cost = cost + 1
            child_key = key(child)
            if seen.get(child_key, 10**9) <= child_cost:
                continue
            seen[child_key] = child_cost
            score = metric(child)
            if score < best:
                best = score
                print("progress", len(seen), child_cost, best, centers(child, 14), centers(child, 11))
            heapq.heappush(heap, (child_cost + score // 3, child_cost, next(serial),
                                  child, child_path))
        active = active_center(node)
        for target in centers(node, 14):
            if target == active:
                continue
            child = node.clone()
            child.step(6, target[1], target[0])
            child_path = path + [(6, target[1], target[0])]
            child_cost = cost + 1
            child_key = key(child)
            if seen.get(child_key, 10**9) <= child_cost:
                continue
            seen[child_key] = child_cost
            heapq.heappush(heap, (child_cost + metric(child) // 3, child_cost, next(serial),
                                  child, child_path))
    print("NOT_FOUND", len(seen), "best", best)


print("run_result", A.run_program("ka59", probe))

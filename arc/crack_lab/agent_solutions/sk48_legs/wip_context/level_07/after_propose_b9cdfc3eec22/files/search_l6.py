import sys
import heapq
import itertools

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

import players
from perception import connected_components


def dense_cost(env):
    pieces = connected_components(env.frame(), colors=(6, 8, 9), min_area=16)
    avatar = next(b for b in pieces if b.color == 6 and b.bbox[0] < 53)
    ar = round(avatar.centroid[0])
    eights = sorted((round(b.centroid[0]), round(b.centroid[1]))
                    for b in pieces if b.color == 8 and b.bbox[0] < 53)
    nines = sorted((round(b.centroid[1]), round(b.centroid[0]))
                   for b in pieces if b.color == 9 and b.bbox[0] < 53)
    d8 = sum(abs(r - gr) + abs(c - 32) for (r, c), gr in zip(eights, (11, 17, 23)))
    d9 = sum(abs(c - gc) + abs(r - ar) for (c, r), gc in zip(nines, (14, 20, 26)))
    return (d8 + d9) // 6


def best_first(env, max_states=12000, max_depth=70):
    root = env.clone()
    serial = itertools.count()
    heap = [(dense_cost(root), 0, next(serial), ())]
    seen = {np.asarray(root.frame()).tobytes()}
    expanded = 0
    while heap and expanded < max_states:
        _, depth, _, path = heapq.heappop(heap)
        node = root.clone()
        for action in path:
            node.step(action)
        expanded += 1
        if node.levels_completed > 5:
            return list(path), expanded
        if depth >= max_depth:
            continue
        for action in (1, 2, 3, 4):
            child = node.clone()
            child.step(action)
            key = np.asarray(child.frame()).tobytes()
            if key in seen:
                continue
            seen.add(key)
            child_path = path + (action,)
            score = depth + 1 + 2 * dense_cost(child)
            heapq.heappush(heap, (score, depth + 1, next(serial), child_path))
    return None, expanded


def search(env):
    for level in range(1, 6):
        getattr(players, f"play_level_{level}")(env)
        assert env.levels_completed == level
    path, expanded = best_first(env)
    print("FOUND", path, "EXPANDED", expanded)


A.run_program("sk48", search)

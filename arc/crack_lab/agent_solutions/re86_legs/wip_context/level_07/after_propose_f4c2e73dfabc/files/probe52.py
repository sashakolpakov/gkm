import heapq
import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

from perception import arr
from probe48 import (
    DOWN, LEFT, NAME, PATH, RIGHT, SPECS, UP, USE,
    bare_frame, visible_shape,
)


TARGETS = SPECS[1][2]
DELTA = {UP: (-3, 0), DOWN: (3, 0), LEFT: (0, -3), RIGHT: (0, 3)}
OPPOSITE = {UP: DOWN, DOWN: UP, LEFT: RIGHT, RIGHT: LEFT}


def shape(node, bare, center):
    return visible_shape(arr(node.frame()), bare, center)


def score(item):
    if item is None:
        return 999
    points = item[2]
    return sum(min(abs(r-x) + abs(c-y) for x, y in points) for r, c in TARGETS)


def key(item):
    if item is None:
        return None
    return item[1], item[0], tuple(sorted(item[2]))


def search(root, bare, max_states=16000, max_depth=42):
    center = (51, 21)
    item = shape(root, bare, center)
    serial = 0
    queue = [(score(item), 0, serial, root.clone(), (), center, item)]
    seen = {key(item): 0}
    best = (score(item), ())
    while queue and len(seen) < max_states:
        _, depth, _, node, path, center, item = heapq.heappop(queue)
        if depth >= max_depth:
            continue
        for action in (UP, DOWN, LEFT, RIGHT):
            if path and action == OPPOSITE[path[-1]]:
                continue
            child = node.clone()
            child.step(action)
            dr, dc = DELTA[action]
            child_center = center[0] + dr, center[1] + dc
            child_item = shape(child, bare, child_center)
            child_key = key(child_item)
            next_depth = depth + 1
            if seen.get(child_key, 999) <= next_depth:
                continue
            seen[child_key] = next_depth
            next_path = path + (action,)
            value = score(child_item)
            if value < best[0]:
                best = value, next_path
                hits = sum(point in child_item[2] for point in TARGETS)
                print("best", value, "hits", hits, "color", child_item[1],
                      "n", len(child_item[2]), len(next_path),
                      "".join(NAME[a] for a in next_path), flush=True)
            if child_item[1] != 11 and all(
                    point in child_item[2] for point in TARGETS):
                print("SOLVED", len(next_path),
                      "".join(NAME[a] for a in next_path),
                      child_item[1], len(child_item[2]), len(seen), flush=True)
                return next_path
            serial += 1
            heapq.heappush(
                queue,
                (next_depth + value, next_depth, serial, child, next_path,
                 child_center, child_item),
            )
    print("FAILED", len(seen), best[0],
          "".join(NAME[a] for a in best[1]), flush=True)


def run(env):
    for action in PATH:
        env.step(action)
    root = env.clone()
    root.step(USE)
    bare = bare_frame(root)
    search(root, bare)


A.run_program("re86", run)

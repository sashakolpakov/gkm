import heapq
import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

from perception import arr
from probe48 import DOWN, LEFT, NAME, PATH, RIGHT, UP, USE


TARGETS = ((30, 45), (48, 39), (48, 51))
DELTA = {UP: (-3, 0), DOWN: (3, 0), LEFT: (0, -3), RIGHT: (0, 3)}
OPPOSITE = {UP: DOWN, DOWN: UP, LEFT: RIGHT, RIGHT: LEFT}


def points(node):
    frame = arr(node.frame())
    return {(int(r), int(c)) for r, c in zip(*((frame == 12).nonzero()))}


def normalized(pts, center):
    return tuple(sorted((r - center[0], c - center[1]) for r, c in pts))


def pattern_score(pts):
    best = (999, None)
    tr, tc = TARGETS[0]
    for row, col in pts:
        shifted = tuple((r - tr + row, c - tc + col) for r, c in TARGETS)
        value = sum(
            min(abs(r-x) + abs(c-y) for x, y in pts)
            for r, c in shifted
        )
        if value < best[0]:
            best = value, shifted
    return best


def search(root, max_states=60000, max_depth=54):
    start_center = (51, 21)
    start_points = points(root)
    serial = 0
    start_value, _ = pattern_score(start_points)
    queue = [(start_value, 0, serial, root.clone(), (), start_center)]
    seen = {(start_center, normalized(start_points, start_center)): 0}
    best = start_value, ()
    while queue and len(seen) < max_states:
        _, depth, _, node, path, center = heapq.heappop(queue)
        if depth >= max_depth:
            continue
        for action in (UP, DOWN, LEFT, RIGHT):
            child = node.clone()
            child.step(action)
            dr, dc = DELTA[action]
            child_center = center[0] + dr, center[1] + dc
            child_points = points(child)
            child_norm = normalized(child_points, child_center)
            state = child_center, child_norm
            next_depth = depth + 1
            if seen.get(state, 999) <= next_depth:
                continue
            seen[state] = next_depth
            next_path = path + (action,)
            value, shifted = pattern_score(child_points)
            if value < best[0]:
                best = value, next_path
                print("best", value, "center", child_center,
                      "bbox", (min(r for r, _ in child_points),
                               min(c for _, c in child_points),
                               max(r for r, _ in child_points),
                               max(c for _, c in child_points)),
                      len(next_path), "".join(NAME[a] for a in next_path),
                      flush=True)
            if value == 0:
                dr = TARGETS[0][0] - shifted[0][0]
                dc = TARGETS[0][1] - shifted[0][1]
                goal_center = child_center[0] + dr, child_center[1] + dc
                print("SOLVED-GEOMETRY", len(next_path),
                      "".join(NAME[a] for a in next_path),
                      "at", child_center, "place", goal_center,
                      "shift", (dr, dc), len(seen), flush=True)
                return next_path, goal_center
            serial += 1
            heapq.heappush(
                queue,
                (next_depth + 2 * value, next_depth, serial, child,
                 next_path, child_center),
            )
    print("FAILED", len(seen), best[0],
          "".join(NAME[a] for a in best[1]), flush=True)


def run(env):
    for action in PATH:
        env.step(action)
    root = env.clone()
    root.step(USE)
    search(root)


A.run_program("re86", run)

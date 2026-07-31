import json
import sys
from heapq import heappop, heappush

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A

from perception import arr


PREFIX = 306
SEGMENTS = (
    [(4, 10), (1, 7), (4, 1), (2, 2), (1, 7), (3, 6),
     (2, 2), (4, 6), (2, 7)],
    [(1, 3), (4, 4), (1, 2), (4, 5), (1, 6), (3, 13),
     (1, 4), (2, 4), (4, 13)],
    [(1, 1), (3, 2), (1, 8), (3, 3), (1, 3), (4, 3),
     (2, 4), (1, 3), (4, 6), (3, 6), (1, 4), (2, 3)],
)


def expand(runs):
    return [action for action, count in runs for _ in range(count)]


def step_many(node, path):
    for action in path:
        node.step(action)


def cursor(grid):
    points = list(zip(*((grid == 0).nonzero())))
    return tuple(int(value) for value in points[0]) if len(points) == 1 else None


def search(root, target_frame, max_depth, max_states=50000):
    target_cursor = cursor(target_frame)
    serial = 0
    start_frame = arr(root.frame())
    start_cursor = cursor(start_frame)
    start_h = (
        abs(start_cursor[0] - target_cursor[0])
        + abs(start_cursor[1] - target_cursor[1])
    ) // 3
    queue = [(start_h, serial, root.clone(), [])]
    seen = {start_frame.tobytes(): 0}
    expanded = 0
    while queue and len(seen) < max_states:
        _, _, node, path = heappop(queue)
        frame = arr(node.frame())
        expanded += 1
        if frame.tobytes() == target_frame.tobytes():
            print("FOUND", len(path), path, len(seen), expanded)
            return path
        if len(path) >= max_depth:
            continue
        for action in (1, 2, 3, 4):
            child = node.clone()
            child.step(action)
            child_frame = arr(child.frame())
            key = child_frame.tobytes()
            depth = len(path) + 1
            if seen.get(key, max_depth + 1) <= depth:
                continue
            seen[key] = depth
            child_cursor = cursor(child_frame)
            if child_cursor is None:
                continue
            distance = (
                abs(child_cursor[0] - target_cursor[0])
                + abs(child_cursor[1] - target_cursor[1])
            ) // 3
            mismatch = int((child_frame != target_frame).sum()) // 64
            serial += 1
            heappush(
                queue,
                (depth + distance + mismatch, serial, child, path + [action]),
            )
        if expanded % 2000 == 0:
            print("PROGRESS", expanded, len(seen), len(queue), len(path), flush=True)
    print("NOT_FOUND", expanded, len(seen), len(queue))
    return None


def probe(env):
    with open("checkpoint.json") as handle:
        full = json.load(handle)["final_path"]
    step_many(env, full[:PREFIX])
    root = env.clone()
    step_many(root, expand(SEGMENTS[0]))
    root.step(5)
    step_many(root, expand(SEGMENTS[1]))
    root.step(5)
    known = expand(SEGMENTS[2])[:-1]
    target = root.clone()
    step_many(target, known)
    print("known", len(known), "cursors", cursor(arr(root.frame())), cursor(arr(target.frame())))
    search(root, arr(target.frame()).copy(), len(known) - 1)


if __name__ == "__main__":
    A.run_program("re86", probe)

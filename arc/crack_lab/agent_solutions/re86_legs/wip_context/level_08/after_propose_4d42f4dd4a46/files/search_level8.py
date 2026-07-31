import json
import sys
from heapq import heappop, heappush

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A

from perception import arr


TARGETS = {
    6: {(45, 6), (45, 21), (54, 6), (54, 21)},
    11: {(39, 9), (39, 15), (57, 9), (57, 15)},
}
TARGET_BBOX = {
    color: (
        min(row for row, _ in points),
        min(col for _, col in points),
        max(row for row, _ in points),
        max(col for _, col in points),
    )
    for color, points in TARGETS.items()
}


def selected_shape(node):
    before = arr(node.frame()).copy()
    cursor_points = list(zip(*((before == 0).nonzero())))
    if len(cursor_points) != 1:
        return None
    center = tuple(int(v) for v in cursor_points[0])
    frames = []
    votes = []
    for action in (1, 2, 3, 4):
        moved = node.clone()
        moved.step(action)
        after = arr(moved.frame())
        frames.append(after)
        for row, col in zip(*((before != after).nonzero())):
            if int(after[row, col]) == 5 and int(before[row, col]) not in (0, 5):
                votes.append(int(before[row, col]))
    if not votes:
        return None
    color = max(set(votes), key=votes.count)
    offsets = set()
    for after in frames:
        for row, col in zip(*((before != after).nonzero())):
            if int(before[row, col]) == color:
                offsets.add((int(row) - center[0], int(col) - center[1]))
    offsets.discard((0, 0))
    return center, color, offsets


def probe(env):
    with open("checkpoint.json") as handle:
        for action in json.load(handle)["final_path"]:
            env.step(action)

    root = env.clone()
    root.step(5)
    for _ in range(22):
        root.step(1)
    root.step(5)
    seed = [
        3, 3, 3, 3, 3, 3, 3, 3, 2, 4, 4, 3,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 1, 3, 3, 1,
    ]
    for action in seed:
        root.step(action)

    candidates = {
        (row, col)
        for row in range(33, 61, 3)
        for col in range(0, 64, 3)
    }

    def heuristic(frame):
        points = list(zip(*((arr(frame) == 0).nonzero())))
        if not points:
            return 100
        row, col = points[0]
        return min((abs(row - r) + abs(col - c)) // 3 for r, c in candidates)

    serial = 0
    queue = [(heuristic(root.frame()), serial, root, [])]
    seen = {arr(root.frame()).tobytes()}
    geometries = {}
    while queue and len(seen) < 6000:
        _, _, node, path = heappop(queue)
        cursor_points = list(zip(*((arr(node.frame()) == 0).nonzero())))
        if not cursor_points:
            continue
        cursor = cursor_points[0]
        if cursor in candidates:
            shape = selected_shape(node)
            if shape is not None:
                center, color, offsets = shape
                rows = [center[0] + row for row, _ in offsets]
                cols = [center[1] + col for _, col in offsets]
                bbox = (min(rows), min(cols), max(rows), max(cols))
                signature = (color, bbox[2] - bbox[0], bbox[3] - bbox[1])
                geometries.setdefault(signature, (center, len(path), path))
                if color == 11 and (bbox[2] - bbox[0], bbox[3] - bbox[1]) == (18, 6):
                    print("FOUND", color, len(seed + path), seed + path)
                    print("shape", center, len(offsets), bbox)
                    return
        if len(path) >= 55:
            continue
        for action in (1, 2, 3, 4):
            child = node.clone()
            child.step(action)
            key = arr(child.frame()).tobytes()
            if key not in seen:
                seen.add(key)
                child_path = path + [action]
                serial += 1
                score = len(child_path) + heuristic(child.frame())
                heappush(queue, (score, serial, child, child_path))
    print("NOT_FOUND", len(seen), "queued", len(queue))
    print("GEOMETRIES", sorted((key, value[:2]) for key, value in geometries.items()))
    for key, value in geometries.items():
        if key[1:] in ((18, 6), (9, 15), (15, 9)):
            print("ROUTE", key, value)


print("RUN", A.run_program("re86", probe))

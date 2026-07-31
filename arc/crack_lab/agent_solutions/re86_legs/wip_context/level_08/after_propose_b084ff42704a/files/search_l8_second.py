import json
import sys
from heapq import heappop, heappush

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A

from perception import arr
from probe_l8_routes import selected_shape, step_many


FIRST = [3] * 8 + [2] + [4] * 2 + [1] + [3] * 6
TARGET_CURSORS = tuple(
    (row, col)
    for row in range(39, 61, 3)
    for col in range(0, 28, 3)
)
TARGET_CENTER_CURSORS = ((48, 12), (48, 15), (51, 12), (51, 15))


def cursor(node):
    points = list(zip(*((arr(node.frame()) == 0).nonzero())))
    if len(points) != 1:
        return None
    return int(points[0][0]), int(points[0][1])


def distance_to_target(center):
    return min(
        (abs(center[0] - row) + abs(center[1] - col)) // 3
        for row, col in TARGET_CENTER_CURSORS
    )


def probe(env):
    with open("checkpoint.json") as handle:
        step_many(env, json.load(handle)["final_path"])
    step_many(env, FIRST)
    print("first", selected_shape(env))
    env.step(5)
    print("second", selected_shape(env))

    serial = 0
    root = env.clone()
    queue = [(distance_to_target(cursor(root)), serial, root, [])]
    seen = {arr(root.frame()).tobytes()}
    expanded = 0
    while queue and len(seen) < 60000:
        _, _, node, path = heappop(queue)
        expanded += 1
        center = cursor(node)
        if node.levels_completed > 7:
            print("FOUND_REWARD", path, len(seen), expanded)
            return
        if center in TARGET_CURSORS:
            shape = selected_shape(node)
            if shape is not None and shape[2:] == (48, (45, 6, 54, 21)):
                print("GEOMETRY", len(path), shape, path)
        if len(path) >= 90 or center is None:
            continue
        for action, inverse in ((1, 2), (2, 1), (3, 4), (4, 3)):
            if path and path[-1] == inverse:
                continue
            child = node.clone()
            child.step(action)
            key = arr(child.frame()).tobytes()
            if key in seen:
                continue
            seen.add(key)
            following = cursor(child)
            if following is None:
                continue
            serial += 1
            child_path = path + [action]
            heappush(
                queue,
                (
                    len(child_path) + distance_to_target(following),
                    serial,
                    child,
                    child_path,
                ),
            )
        if expanded % 500 == 0:
            print("PROGRESS", expanded, len(seen), len(queue), len(path))
    print("NOT_FOUND", expanded, len(seen), len(queue))


print("RUN", A.run_program("re86", probe))

import json
import sys
from heapq import heappop, heappush

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A

from perception import arr
from probe_l8_routes import selected_shape, step_many


FIRST = [3] * 8 + [2] + [4] * 2 + [1] + [3] * 6
DEFORM_SECOND = [2] + [3] * 4 + [2]
TARGET_BBOX = (45, 6, 54, 21)


def cursor(node):
    points = list(zip(*((arr(node.frame()) == 0).nonzero())))
    return tuple(int(value) for value in points[0]) if len(points) == 1 else None


def target_perimeter(node):
    grid = arr(node.frame())
    row0, col0, row1, col1 = TARGET_BBOX
    values = (
        list(grid[row0, col0 : col1 + 1])
        + list(grid[row1, col0 : col1 + 1])
        + list(grid[row0 + 1 : row1, col0])
        + list(grid[row0 + 1 : row1, col1])
    )
    return len(set(int(value) for value in values)) == 1


def probe(env):
    with open("checkpoint.json") as handle:
        step_many(env, json.load(handle)["final_path"])
    step_many(env, FIRST)
    env.step(5)
    step_many(env, DEFORM_SECOND)

    serial = 0
    root = env.clone()
    queue = [(9, serial, root, [])]
    seen = {arr(root.frame()).tobytes()}
    expanded = 0
    while queue and len(seen) < 30000:
        _, _, node, path = heappop(queue)
        expanded += 1
        center = cursor(node)
        if node.levels_completed > 7 or target_perimeter(node):
            print(
                "FOUND",
                path,
                "cursor",
                center,
                "shape",
                selected_shape(node),
                "level",
                node.levels_completed,
                "seen",
                len(seen),
                "expanded",
                expanded,
            )
            return
        if len(path) >= 45 or center is None:
            continue
        for action in (1, 2, 3, 4):
            child = node.clone()
            child.step(action)
            key = arr(child.frame()).tobytes()
            if key in seen:
                continue
            seen.add(key)
            following = cursor(child)
            if following is None:
                continue
            distance = (
                abs(following[0] - 50) + abs(following[1] - 14)
            ) // 3
            serial += 1
            child_path = path + [action]
            heappush(
                queue,
                (len(child_path) + distance, serial, child, child_path),
            )
        if expanded % 1000 == 0:
            print("PROGRESS", expanded, len(seen), len(queue), len(path))
    print("NOT_FOUND", expanded, len(seen), len(queue))


if __name__ == "__main__":
    print("RUN", A.run_program("re86", probe))

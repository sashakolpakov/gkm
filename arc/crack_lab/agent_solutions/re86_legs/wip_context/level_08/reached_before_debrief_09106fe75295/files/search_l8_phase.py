import json
import sys
from collections import deque

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A

from perception import arr
from probe_l8_routes import SEED, selected_shape, step_many


def cursor(node):
    points = list(zip(*((arr(node.frame()) == 0).nonzero())))
    return tuple(int(value) for value in points[0]) if len(points) == 1 else None


def colored_rectangle(frame, height=19, width=7, color=11):
    grid = arr(frame)
    for row0 in range(65 - height):
        row1 = row0 + height - 1
        for col0 in range(65 - width):
            col1 = col0 + width - 1
            values = (
                list(grid[row0, col0 : col1 + 1])
                + list(grid[row1, col0 : col1 + 1])
                + list(grid[row0 + 1 : row1, col0])
                + list(grid[row0 + 1 : row1, col1])
            )
            if all(int(value) == color for value in values):
                return row0, col0, row1, col1
    return None


def probe(env):
    with open("checkpoint.json") as handle:
        step_many(env, json.load(handle)["final_path"])
    step_many(env, SEED)

    queue = deque([(env.clone(), [])])
    seen = {arr(env.frame()).tobytes()}
    expanded = 0
    while queue and len(seen) < 20000:
        node, path = queue.popleft()
        expanded += 1
        center = cursor(node)
        if (
            path
            and center is not None
            and center[0] > 32
            and int((arr(node.frame()) == 11).sum()) >= 50
        ):
            rectangle = colored_rectangle(node.frame())
            print(
                "FOUND",
                path,
                "cursor",
                center,
                "rectangle",
                rectangle,
                "shape",
                selected_shape(node),
                "seen",
                len(seen),
                "expanded",
                expanded,
            )
            return
        if len(path) >= 35:
            continue
        for action in (1, 2, 3, 4):
            child = node.clone()
            child.step(action)
            key = arr(child.frame()).tobytes()
            if key not in seen:
                seen.add(key)
                queue.append((child, path + [action]))
        if expanded % 1000 == 0:
            print("PROGRESS", expanded, len(seen), len(queue), len(path))
    print("NOT_FOUND", expanded, len(seen), len(queue))


if __name__ == "__main__":
    print("RUN", A.run_program("re86", probe))

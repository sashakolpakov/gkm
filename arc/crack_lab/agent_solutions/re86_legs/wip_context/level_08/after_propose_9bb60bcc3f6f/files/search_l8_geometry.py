import json
import sys
from collections import deque

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A

from perception import arr
from probe_l8_routes import selected_shape, step_many


FIRST = [3] * 8 + [2] + [4] * 2 + [1] + [3] * 6


def exact_rectangle(frame, height, width):
    grid = arr(frame)
    allowed = {1, 6, 8, 9, 10, 11, 12, 14}
    for row0 in range(65 - height):
        row1 = row0 + height - 1
        for col0 in range(65 - width):
            col1 = col0 + width - 1
            perimeter = (
                list(grid[row0, col0 : col1 + 1])
                + list(grid[row1, col0 : col1 + 1])
                + list(grid[row0 + 1 : row1, col0])
                + list(grid[row0 + 1 : row1, col1])
            )
            color = int(perimeter[0])
            if color in allowed and all(int(value) == color for value in perimeter):
                return color, (row0, col0, row1, col1)
    return None


def probe(env):
    with open("checkpoint.json") as handle:
        step_many(env, json.load(handle)["final_path"])
    step_many(env, FIRST)
    env.step(5)

    queue = deque([(env.clone(), [])])
    seen = {arr(env.frame()).tobytes()}
    expanded = 0
    while queue and len(seen) < 30000:
        node, path = queue.popleft()
        expanded += 1
        rectangle = exact_rectangle(node.frame(), 10, 16)
        if rectangle is not None:
            print(
                "FOUND",
                rectangle,
                "shape",
                selected_shape(node),
                "path",
                path,
                "seen",
                len(seen),
                "expanded",
                expanded,
            )
            return
        if len(path) >= 32:
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

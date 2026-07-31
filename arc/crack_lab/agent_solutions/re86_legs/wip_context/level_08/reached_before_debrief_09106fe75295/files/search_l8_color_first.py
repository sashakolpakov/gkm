import json
import sys
from heapq import heappop, heappush

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A

from perception import arr
from probe_l8_routes import SEED, selected_shape, step_many


TARGET = (
    [(39, col) for col in range(9, 16)]
    + [(57, col) for col in range(9, 16)]
    + [(row, 9) for row in range(40, 57)]
    + [(row, 15) for row in range(40, 57)]
)


def cursor(node):
    points = list(zip(*((arr(node.frame()) == 0).nonzero())))
    return tuple(int(value) for value in points[0]) if len(points) == 1 else None


def progress(node):
    grid = arr(node.frame())
    correct = sum(int(grid[row, col]) == 11 for row, col in TARGET)
    painted = int((grid == 11).sum()) >= 50
    return correct, painted


def probe(env):
    with open("checkpoint.json") as handle:
        step_many(env, json.load(handle)["final_path"])
    step_many(env, SEED)
    print("root", selected_shape(env), progress(env))

    serial = 0
    root = env.clone()
    center = cursor(root)
    queue = [(
        (abs(center[0] - 48) + abs(center[1] - 12)) // 3,
        serial,
        root,
        [],
    )]
    seen = {arr(root.frame()).tobytes()}
    expanded = 0
    while queue and len(seen) < 50000:
        _, _, node, path = heappop(queue)
        expanded += 1
        center = cursor(node)
        correct, painted = progress(node)
        if correct == 48:
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
        if len(path) >= 90 or center is None:
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
            child_correct, child_painted = progress(child)
            distance = (
                abs(following[0] - 48) + abs(following[1] - 12)
            ) // 3
            child_path = path + [action]
            score = (
                len(child_path)
                + distance
                + (0 if child_painted else 40)
                - 2 * child_correct
            )
            serial += 1
            heappush(queue, (score, serial, child, child_path))
        if expanded % 1000 == 0:
            print(
                "PROGRESS",
                expanded,
                len(seen),
                len(queue),
                len(path),
                correct,
                painted,
            )
    print("NOT_FOUND", expanded, len(seen), len(queue))


if __name__ == "__main__":
    print("RUN", A.run_program("re86", probe))

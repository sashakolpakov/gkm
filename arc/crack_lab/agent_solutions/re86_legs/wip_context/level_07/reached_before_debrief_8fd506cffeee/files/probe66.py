import json
import sys
from collections import defaultdict

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

from perception import DOWN, LEFT, RIGHT, UP, USE, arr


PATH = json.load(open("checkpoint.json"))["final_path"]
TARGETS = ((30, 45), (48, 39), (48, 51))
BASE = (UP,) * 4 + (RIGHT,) * 2 + (UP,)
NAME = {UP: "U", DOWN: "D", LEFT: "L", RIGHT: "R"}


def shape(node):
    frame = arr(node.frame())
    return {
        (int(row), int(col))
        for row, col in zip(*((frame == 12).nonzero()))
    }


def pattern(points):
    tr, tc = TARGETS[0]
    best = (999, None)
    for row, col in points:
        shifted = tuple((r - tr + row, c - tc + col) for r, c in TARGETS)
        value = sum(
            min(abs(r - sr) + abs(c - sc) for sr, sc in points)
            for r, c in shifted
        )
        best = min(best, (value, shifted))
    return best


def brief(node):
    points = shape(node)
    by_row = defaultdict(list)
    for row, col in points:
        by_row[row].append(col)
    bbox = (
        min(by_row),
        min(col for cols in by_row.values() for col in cols),
        max(by_row),
        max(col for cols in by_row.values() for col in cols),
    )
    extrema = tuple(
        (row, min(by_row[row]), max(by_row[row]), len(by_row[row]))
        for row in (min(by_row), max(by_row))
    )
    return pattern(points), bbox, extrema, len(points)


def run(env):
    for action in PATH:
        env.step(action)
    root = env.clone()
    root.step(USE)
    for action in BASE:
        root.step(action)
    print("BASE", brief(root), flush=True)
    for rights in range(1, 5):
        node = root.clone()
        for _ in range(rights):
            node.step(RIGHT)
        print("R" + str(rights), brief(node), flush=True)
    routes = []
    for rights in range(1, 9):
        routes.append((RIGHT,) * rights)
        for downs in range(1, 7):
            routes.append((RIGHT,) * rights + (DOWN,) * downs)
            routes.append(
                (RIGHT,) * rights + (DOWN,) * downs + (LEFT,) * rights
            )
    best = (brief(root)[0][0], ())
    for route in routes:
        node = root.clone()
        for action in route:
            node.step(action)
        value = brief(node)
        if value[0][0] < best[0]:
            best = value[0][0], route
            print(
                "BEST",
                "".join(NAME[action] for action in route),
                value,
                flush=True,
            )
        if value[0][0] == 0:
            print(
                "SOLVED",
                "".join(NAME[action] for action in route),
                value,
                flush=True,
            )
            return
    print("DONE", best[0], flush=True)


A.run_program("re86", run)

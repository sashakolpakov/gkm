import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

from perception import DOWN, RIGHT, UP, USE, arr
from probe48 import bare_frame, visible_shape


PATH = json.load(open("checkpoint.json"))["final_path"]
TARGETS = ((30, 45), (48, 39), (48, 51))
NAME = {UP: "U", DOWN: "D", RIGHT: "R"}


DELTA = {UP: (-3, 0), DOWN: (3, 0), RIGHT: (0, 3)}


def pattern_score(node, bare, center):
    item = visible_shape(arr(node.frame()), bare, center)
    shape = item[2] - {center}
    best = (999, None)
    tr, tc = TARGETS[0]
    for row, col in shape:
        shifted = tuple((r - tr + row, c - tc + col) for r, c in TARGETS)
        value = sum(
            min(abs(r - sr) + abs(c - sc) for sr, sc in shape)
            for r, c in shifted
        )
        if value < best[0]:
            best = value, shifted
    return best[0], best[1], len(shape)


def replay(root, route, center):
    node = root.clone()
    for action in route:
        node.step(action)
        dr, dc = DELTA[action]
        center = center[0] + dr, center[1] + dc
    return node, center


def run(env):
    for action in PATH:
        env.step(action)
    root = env.clone()
    root.step(USE)
    bare = bare_frame(root)
    start_center = (51, 21)

    old = (UP,) * 3 + (RIGHT,) * 4 + (UP,) * 8 + (RIGHT,) * 9 + (DOWN,) * 10
    node = root.clone()
    center = start_center
    best = (999, ())
    for index, action in enumerate(old, 1):
        node.step(action)
        dr, dc = DELTA[action]
        center = center[0] + dr, center[1] + dc
        value = pattern_score(node, bare, center)
        if value[0] < best[0]:
            best = value[0], old[:index]
            print(
                "TRACE",
                index,
                NAME[action],
                value,
                flush=True,
            )

    for middle_ups in range(3, 11):
        for rights in range(3, 11):
            for downs in range(1, 13):
                route = (
                    (UP,) * 3
                    + (RIGHT,) * 4
                    + (UP,) * middle_ups
                    + (RIGHT,) * rights
                    + (DOWN,) * downs
                )
                candidate, center = replay(root, route, start_center)
                value = pattern_score(candidate, bare, center)
                if value[0] < best[0]:
                    best = value[0], route
                    print(
                        "BEST",
                        value,
                        (middle_ups, rights, downs),
                        flush=True,
                    )
                if value[0] == 0:
                    print(
                        "SOLVED",
                        (middle_ups, rights, downs),
                        value,
                        flush=True,
                    )
                    return
    print("DONE", best[0], len(best[1]), flush=True)


A.run_program("re86", run)

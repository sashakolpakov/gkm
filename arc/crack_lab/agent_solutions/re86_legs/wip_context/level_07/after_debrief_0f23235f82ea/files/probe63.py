import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

from perception import DOWN, LEFT, RIGHT, UP, USE, arr


PATH = json.load(open("checkpoint.json"))["final_path"]
TARGETS = ((30, 45), (48, 39), (48, 51))
NAME = {UP: "U", DOWN: "D", LEFT: "L", RIGHT: "R"}
NEAR = (RIGHT,) * 4 + (UP,) * 3 + (RIGHT,) * 4


def outline(node, center):
    frame = arr(node.frame())
    points = {
        (int(row), int(col))
        for row, col in zip(*((frame == 12).nonzero()))
    }
    if 0 <= center[0] < 63 and 0 <= center[1] < 64 and int(frame[center]) == 0:
        points.add(center)
    return points


def summary(node, center):
    points = outline(node, center)
    distances = tuple(
        min(abs(row - tr) + abs(col - tc) for row, col in points)
        for tr, tc in TARGETS
    )
    bbox = (
        min(row for row, _ in points),
        min(col for _, col in points),
        max(row for row, _ in points),
        max(col for _, col in points),
    )
    return sum(distances), distances, bbox, len(points)


def run(env):
    for action in PATH:
        env.step(action)
    root = env.clone()
    root.step(USE)
    center = (51, 21)
    for action in NEAR:
        root.step(action)
        dr, dc = {
            UP: (-3, 0), DOWN: (3, 0), LEFT: (0, -3), RIGHT: (0, 3)
        }[action]
        center = center[0] + dr, center[1] + dc
    print("NEAR", center, summary(root, center), flush=True)
    for tag, route in (
        ("L2U", (LEFT,) * 2 + (UP,)),
        ("L2U2", (LEFT,) * 2 + (UP,) * 2),
        ("L3U2R3", (LEFT,) * 3 + (UP,) * 2 + (RIGHT,) * 3),
        ("L4U3R4D", (LEFT,) * 4 + (UP,) * 3 + (RIGHT,) * 4 + (DOWN,)),
    ):
        node = root.clone()
        route_center = center
        for action in route:
            node.step(action)
            dr, dc = {
                UP: (-3, 0), DOWN: (3, 0), LEFT: (0, -3), RIGHT: (0, 3)
            }[action]
            route_center = route_center[0] + dr, route_center[1] + dc
        print(tag, route_center, summary(node, route_center), flush=True)

    best = summary(root, center)[0], ()
    for lefts in range(1, 8):
        for ups in range(1, 8):
            loops = (
                (LEFT,) * lefts + (UP,) * ups
                + (RIGHT,) * lefts + (DOWN,) * ups,
                (LEFT,) * lefts + (UP,) * ups
                + (DOWN,) * ups + (RIGHT,) * lefts,
                (UP,) * ups + (LEFT,) * lefts
                + (DOWN,) * ups + (RIGHT,) * lefts,
                (LEFT,) * lefts + (DOWN,) * ups
                + (RIGHT,) * lefts + (UP,) * ups,
            )
            for loop in loops:
                node = root.clone()
                for action in loop:
                    node.step(action)
                value = summary(node, center)
                rank = value[0]
                if rank < best[0]:
                    best = rank, loop
                    print(
                        "BEST",
                        value,
                        "".join(NAME[action] for action in loop),
                        flush=True,
                    )
                if rank == 0:
                    print(
                        "SOLVED",
                        "".join(NAME[action] for action in loop),
                        value,
                        flush=True,
                    )
                    return
    for lefts in range(2, 7):
        for ups in range(1, 7):
            for rights in range(1, 9):
                for downs in range(0, 7):
                    route = (
                        (LEFT,) * lefts
                        + (UP,) * ups
                        + (RIGHT,) * rights
                        + (DOWN,) * downs
                    )
                    node = root.clone()
                    route_center = center
                    for action in route:
                        node.step(action)
                        dr, dc = {
                            UP: (-3, 0),
                            DOWN: (3, 0),
                            LEFT: (0, -3),
                            RIGHT: (0, 3),
                        }[action]
                        route_center = route_center[0] + dr, route_center[1] + dc
                    value = summary(node, route_center)
                    rank = value[0]
                    if rank < best[0]:
                        best = rank, route
                        print(
                            "BEST",
                            value,
                            route_center,
                            "".join(NAME[action] for action in route),
                            flush=True,
                        )
                    if rank == 0:
                        print(
                            "SOLVED",
                            "".join(NAME[action] for action in route),
                            value,
                            route_center,
                            flush=True,
                        )
                        return
    for ups in range(1, 7):
        for lefts in range(1, 7):
            for downs in range(0, 9):
                for rights in range(0, 9):
                    route = (
                        (UP,) * ups
                        + (LEFT,) * lefts
                        + (DOWN,) * downs
                        + (RIGHT,) * rights
                    )
                    node = root.clone()
                    route_center = center
                    for action in route:
                        node.step(action)
                        dr, dc = {
                            UP: (-3, 0),
                            DOWN: (3, 0),
                            LEFT: (0, -3),
                            RIGHT: (0, 3),
                        }[action]
                        route_center = route_center[0] + dr, route_center[1] + dc
                    value = summary(node, route_center)
                    rank = value[0]
                    if rank < best[0]:
                        best = rank, route
                        print(
                            "BEST",
                            value,
                            route_center,
                            "".join(NAME[action] for action in route),
                            flush=True,
                        )
                    if rank == 0:
                        print(
                            "SOLVED",
                            "".join(NAME[action] for action in route),
                            value,
                            route_center,
                            flush=True,
                        )
                        return
    print(
        "DONE",
        best[0],
        "".join(NAME[action] for action in best[1]),
        flush=True,
    )


A.run_program("re86", run)

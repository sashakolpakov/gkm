import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from perception import arr, connected_components, frame_delta


CONTROLS = {
    "A<": (6, 4, 51), "A>": (6, 10, 51), "B": (6, 17, 51),
    "C<": (6, 26, 51), "C>": (6, 32, 51), "D": (6, 39, 51),
    "F<": (6, 4, 58), "F>": (6, 10, 58), "G": (6, 17, 58),
    "H<": (6, 26, 58), "H>": (6, 32, 58),
    "E<": (6, 54, 51), "E>": (6, 60, 51), "I": (6, 60, 58),
}
ROUTE = (
    "F<", "A>", "B", "B", "A>", "F>", "A>", "C>", "A>", "C>",
    "A>", "D",
    "H>", "H>", "H>", "H>", "H>",
    "H<", "H<", "H<", "H<", "H<",
    "C<", "C<", "C<",
    "F>", "F>", "F>", "F>",
    "D", "D",
    "C>", "C>", "C>", "C>",
    "A>", "A>", "A>", "D", "C>", "C>", "C>",
    "H>", "H>", "H>", "H>", "H>", "H>", "H>", "H>",
)
STAGE = ("H<",) * 5 + ("F>",) * 2 + ("C<", "F>") * 6
RINGS = {
    (6, 22), (7, 21), (7, 23), (8, 22),
    (15, 25), (16, 24), (16, 26), (17, 25),
}


def points(frame, color):
    grid = arr(frame)[:42]
    return tuple((int(r), int(c)) for r, c in zip(*((grid == color).nonzero())))


def joint(frame, first, second):
    one, two = points(frame, first), points(frame, second)
    if not one or not two:
        return (99, 99)
    a, b = min(
        ((p, q) for p in one for q in two),
        key=lambda pair: abs(pair[0][0] - pair[1][0])
        + abs(pair[0][1] - pair[1][1]),
    )
    return ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2)


def summary(env):
    marker = tuple(p for p in points(env.frame(), 13) if p not in RINGS)
    joints = tuple(
        joint(env.frame(), first, second)
        for first, second in ((11, 14), (14, 9), (9, 12))
    )
    bodies = tuple(
        (b.color, b.bbox) for b in connected_components(
            env.frame(), colors={9, 11, 12, 14}, min_area=4
        ) if b.bbox[0] < 42
    )
    return joints, marker, bodies


def run(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)
    for name in ROUTE:
        env.step(*CONTROLS[name])
    print("waypoint", summary(env))
    for name in STAGE:
        env.step(*CONTROLS[name])
    print("staged", summary(env), "level", env.levels_completed)
    for name, action in CONTROLS.items():
        node = env.clone()
        states = []
        for _ in range(8):
            before = node.frame()
            node.step(*action)
            changed = frame_delta(before, node.frame())["count"]
            states.append((changed, summary(node)[:2]))
            if changed == 0 or node.levels_completed > 6:
                break
        print(name, states, "level", node.levels_completed)
    for shift in range(3):
        root = env.clone()
        for _ in range(shift):
            root.step(*CONTROLS["C<"])
        for turn in ("B", "D", "G"):
            node = root.clone()
            before = node.frame()
            node.step(*CONTROLS[turn])
            print(
                "context", shift, turn,
                frame_delta(before, node.frame())["count"], summary(node)[:2],
            )
    for prefix in (
        ("H<",) * 3,
        ("C<",) + ("H<",) * 3,
        ("F<",) + ("H<",) * 3,
    ):
        root = env.clone()
        for name in prefix:
            root.step(*CONTROLS[name])
        for turn in ("B", "D", "G"):
            node = root.clone()
            before = node.frame()
            node.step(*CONTROLS[turn])
            if node.levels_completed > 6:
                print(
                    "turn", prefix, turn,
                    frame_delta(before, node.frame())["count"], "WIN",
                    node.levels_completed,
                )
                continue
            print(
                "turn", prefix, turn,
                frame_delta(before, node.frame())["count"], summary(node)[:2],
            )
    node = env.clone()
    for name in ("C<",) + ("H<",) * 3:
        node.step(*CONTROLS[name])
    for count in range(1, 6):
        before = node.frame()
        node.step(*CONTROLS["D"])
        print(
            "D-cycle", count, frame_delta(before, node.frame())["count"],
            summary(node), "level", node.levels_completed,
        )
    node = env.clone()
    for name in ("C<",) + ("H<",) * 3 + ("D",) * 2:
        node.step(*CONTROLS[name])
    for count in range(1, 6):
        before = node.frame()
        node.step(*CONTROLS["G"])
        print(
            "G-cycle", count, frame_delta(before, node.frame())["count"],
            summary(node), "level", node.levels_completed,
        )
    for turns in range(1, 4):
        root = env.clone()
        for name in ("C<",) + ("H<",) * 3 + ("D",) * turns:
            root.step(*CONTROLS[name])
        for turn in ("B", "G"):
            node = root.clone()
            before = node.frame()
            node.step(*CONTROLS[turn])
            print(
                "nested", turns, turn,
                frame_delta(before, node.frame())["count"], summary(node),
            )
    root = env.clone()
    for name in ("C<",) + ("H<",) * 3 + ("D",) * 2:
        root.step(*CONTROLS[name])
    for name in ("C<", "C>", "F<", "F>", "H<", "H>"):
        node = root.clone()
        states = []
        for _ in range(10):
            before = node.frame()
            node.step(*CONTROLS[name])
            changed = frame_delta(before, node.frame())["count"]
            states.append((changed, summary(node)[:2]))
            if changed == 0 or node.levels_completed > 6:
                break
        print("north", name, states, "level", node.levels_completed)
    node = env.clone()
    finish = (
        ("E<",) * 2 + ("I",) * 2 + ("E>",) * 2
        + ("C<",) + ("H<",) * 3 + ("D",) * 2
        + ("C>",) * 7 + ("H>",) * 6
    )
    finish_trace = []
    for name in finish:
        before = node.frame()
        node.step(*CONTROLS[name])
        finish_trace.append((name, frame_delta(before, node.frame())["count"]))
        if node.levels_completed > 6:
            break
    print("finish", finish_trace, summary(node)[:2], "level", node.levels_completed)


arena.run_program("s5i5", run)

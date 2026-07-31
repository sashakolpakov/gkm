import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from perception import (
    ACTION_NAME,
    action_deltas,
    arr,
    bounded_bfs,
    color_counts,
    connected_components,
)


CHARS = {0: ".", 1: " ", 2: "W", 3: "d", 4: "B", 7: "-", 9: "x", 14: "A"}


def draw(frame, r0=20, r1=58, c0=12, c1=52):
    for r, row in enumerate(frame[r0:r1, c0:c1], r0):
        print(f"{r:02d}", "".join(CHARS[int(v)] for v in row))


def brief(frame):
    blobs = connected_components(frame, colors=(1, 2, 4, 7, 9, 14), min_area=4)
    return [
        (b.color, b.bbox, b.area)
        for b in blobs
        if b.area < 3000
    ]


def state(env):
    items = brief(env.frame())
    avatar = [item for item in items if item[0] == 14]
    boxes = [item for item in items if item[0] == 4]
    return (
        "r",
        env.levels_completed,
        "A",
        avatar,
        "B",
        boxes,
        "counts",
        color_counts(env.frame()),
    )


def pieces(env):
    return [
        (b.color, b.bbox, b.area)
        for b in connected_components(
            env.frame(), colors=(0, 3, 4, 9, 14), min_area=4
        )
        if b.bbox[0] < 60
    ]


def try_path(env, label, actions):
    clone = env.clone()
    print("PATH", label, "start", state(clone))
    for action in actions:
        clone.step(action)
        print(" ", ACTION_NAME[action], state(clone))


def focused(env):
    tests = (
        ("two-right", [4, 4]),
        ("pickup", [1, 1, 5]),
        ("carry-into-wall", [1, 1, 5, 1, 1]),
        ("drop-on-wall", [1, 1, 5, 1, 1, 5]),
        (
            "fill-three-slots",
            [
                1, 1, 5, 1, 1, 5,
                2,
                3, 3, 3, 3, 3,
                1, 1,
                4, 5,
                4, 4, 4, 5,
                2,
                4, 4, 4, 4, 4,
                1, 5, 2, 3, 3, 5,
            ],
        ),
        ("drop-at-wall", [1, 1, 5, 1, 5]),
        ("carry-to-left", [1, 1, 5, 3, 3, 3, 3, 1, 1]),
        ("drop-at-left", [1, 1, 5, 3, 3, 3, 3, 1, 1, 5]),
        (
            "connect-three",
            [
                1, 1, 5,
                3, 3, 3, 3, 1, 1, 5,
                4, 4, 4, 4, 4, 4, 4, 1, 1, 5,
                2, 2,
                3, 3, 3, 3, 3, 3, 5,
            ],
        ),
    )
    for label, actions in tests:
        clone = env.clone()
        for action in actions:
            if clone.terminal():
                break
            clone.step(action)
        print("FOCUS", label, "reward", clone.levels_completed, pieces(clone))
        if label in (
            "carry-into-wall",
            "drop-on-wall",
            "fill-three-slots",
            "drop-at-wall",
            "carry-to-left",
            "drop-at-left",
            "connect-three",
        ):
            draw(clone.frame(), 20, 48, 12, 52)


def inspect(env):
    if len(sys.argv) > 1 and sys.argv[1] == "search":
        path = bounded_bfs(
            env,
            lambda node, _: node.levels_completed > env.levels_completed,
            key_fn=lambda node: arr(node.frame())[:-1].tobytes(),
            max_states=5000,
            max_depth=30,
        )
        print("SEARCH", path)
        return
    if len(sys.argv) > 1 and sys.argv[1] == "focus":
        focused(env)
        return
    print("actions", env.actions)
    print("counts", color_counts(env.frame()))
    print("objects", brief(env.frame()))
    draw(env.frame())
    for action, delta in action_deltas(env).items():
        clone = env.clone()
        clone.step(action)
        print(
            ACTION_NAME[action],
            "delta",
            (delta["count"], delta["bbox"]),
            "reward",
            clone.levels_completed,
            "objects",
            brief(clone.frame()),
        )
    for label, actions in (
        ("toward-near-box", [1, 1, 1, 1]),
        ("use-near-box", [1, 1, 1, 5]),
        ("use-facing-away", [1, 1, 1, 2, 5]),
        ("push-and-use", [1, 1, 1, 1, 5]),
        (
            "three-devices",
            [1, 1, 5, 3, 3, 3, 3, 1, 1, 5, 4, 4, 4, 4, 4, 4, 4, 1, 5],
        ),
    ):
        try_path(env, label, actions)


arena.run_program("wa30", inspect)

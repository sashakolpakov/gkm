"""Verify the cargo's top-corridor terminal and subsequent avatar route."""
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

import perception
import solve
from probe_l6_right import (
    MAIN,
    SELECTOR,
    avatar_position,
    enter_right,
    movement_reach,
    scan_controls,
)


UP = (6, 50, 34)
LEFT = (6, 46, 36)
RIGHT = (6, 54, 36)
CARGO_TOP_PATH = [
    1, UP,
    2, 4, RIGHT,
    3, 1, UP,
    2, 1, UP,
    2, 3, LEFT,
    4, 3, LEFT,
    4, 1, UP,
    2, 1, UP,
    2, 1, UP,
]


def ring(env):
    return [
        (blob.bbox, blob.area)
        for blob in perception.connected_components(
            env.frame(), colors=(8,), min_area=20
        )
        if blob.bbox[1] < 32
    ]


def observe(env):
    solve.solve(env)
    base_level = env.levels_completed
    node = enter_right(env, 3)
    for action in CARGO_TOP_PATH:
        if isinstance(action, tuple):
            node.step(*action)
        else:
            node.step(action)
    print(
        "CARGO_TOP", ring(node), avatar_position(node),
        node.levels_completed - base_level,
    )
    print("CARGO_TOP_CONTROLS", scan_controls(node))
    node.step(2)
    node.step(*MAIN)
    print("CARGO_TOP_HUB", avatar_position(node), node.levels_completed)
    hub = node.clone()
    for selector_offset in range(4):
        branch = hub.clone()
        for _ in range(selector_offset):
            branch.step(*SELECTOR)
        branch.step(*MAIN)
        print(
            "CARGO_TOP_DEST", selector_offset,
            avatar_position(branch), branch.levels_completed,
        )
    for _ in range(3):
        node.step(*SELECTOR)
    node.step(*MAIN)
    print("CARGO_TOP_AVATAR", avatar_position(node), node.levels_completed)
    reached, win = movement_reach(node)
    print(
        "CARGO_TOP_AVATAR_REACH",
        sorted(position for position in reached if position is not None),
        "WIN", win,
    )
    simultaneous = node.clone()
    simultaneous.step(4)
    simultaneous.step(*MAIN)
    print(
        "CARGO_TOP_SIMULTANEOUS_MAIN",
        avatar_position(simultaneous),
        simultaneous.levels_completed,
    )
    top_main_wins = []
    for position, path in reached.items():
        if position is None:
            continue
        branch = node.clone()
        for action in path:
            branch.step(action)
        branch.step(*MAIN)
        if branch.levels_completed > base_level:
            top_main_wins.append((position, path))
    print("CARGO_TOP_ALL_MAIN_WINS", top_main_wins)


if __name__ == "__main__":
    arena.run_program("dc22", observe)

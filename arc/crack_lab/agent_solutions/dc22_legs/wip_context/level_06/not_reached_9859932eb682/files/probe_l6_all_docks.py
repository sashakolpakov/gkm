"""Visit lower, middle, and upper cargo docks before the avatar endpoint."""
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

import solve
from probe_l6_cargo_top import CARGO_TOP_PATH
from probe_l6_right import (
    MAIN,
    SELECTOR,
    avatar_position,
    enter_right,
    movement_reach,
)


DOWN = (6, 50, 40)
UP = (6, 50, 34)


def observe(env):
    solve.solve(env)
    base_level = env.levels_completed
    node = enter_right(env, 3)
    node.step(2)
    node.step(*DOWN)
    print("ALL_DOCKS_LOWER", node.levels_completed - base_level)
    node.step(1)
    node.step(1)
    node.step(*UP)
    node.step(2)
    for index, action in enumerate(CARGO_TOP_PATH, start=1):
        if isinstance(action, tuple):
            node.step(*action)
        else:
            node.step(action)
        if node.levels_completed > base_level:
            print("ALL_DOCKS_CARGO_WIN", index)
            return
    node.step(2)
    node.step(*MAIN)
    for _ in range(3):
        node.step(*SELECTOR)
    node.step(*MAIN)
    print("ALL_DOCKS_TOP", avatar_position(node), node.levels_completed)
    reached, win = movement_reach(node)
    print("ALL_DOCKS_TOP_WIN", win)
    committed = enter_right(env, 3)
    committed.step(2)
    committed.step(*DOWN)
    committed.step(*MAIN)
    committed.step(1)
    committed.step(1)
    committed.step(*UP)
    committed.step(2)
    for index, action in enumerate(CARGO_TOP_PATH, start=1):
        if isinstance(action, tuple):
            committed.step(*action)
        else:
            committed.step(action)
        if index == 17:
            committed.step(*MAIN)
    committed.step(*MAIN)
    print(
        "ALL_DOCKS_COMMITTED_CARGO",
        committed.levels_completed - base_level,
    )
    committed.step(2)
    committed.step(*MAIN)
    for _ in range(3):
        committed.step(*SELECTOR)
    committed.step(*MAIN)
    committed_reach, committed_win = movement_reach(committed)
    print(
        "ALL_DOCKS_COMMITTED_TOP", avatar_position(committed),
        committed.levels_completed, committed_win,
    )


arena.run_program("dc22", observe)

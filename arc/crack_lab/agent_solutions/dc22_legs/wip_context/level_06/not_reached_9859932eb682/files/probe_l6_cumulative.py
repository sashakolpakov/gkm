"""Test cumulative visits before routing the cargo to its terminal."""
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
)


def observe(env):
    solve.solve(env)
    base_level = env.levels_completed
    node = enter_right(env, 0)
    for action in (1, 1, 4, 4):
        node.step(action)
    for action in (3, 3, 2, 2):
        node.step(action)
    node.step(*MAIN)
    print("CUMULATIVE_HUB1", avatar_position(node), node.levels_completed)
    node.step(*SELECTOR)
    node.step(*SELECTOR)
    node.step(*MAIN)
    print("CUMULATIVE_TOP", avatar_position(node), node.levels_completed)
    for action in (4, 4, 4, 2, 2, 2, 1, 1, 1, 3, 3, 3):
        node.step(action)
    node.step(*MAIN)
    print("CUMULATIVE_HUB2", avatar_position(node), node.levels_completed)
    node.step(*SELECTOR)
    node.step(*MAIN)
    print("CUMULATIVE_RIGHT3", avatar_position(node), node.levels_completed)
    for index, action in enumerate(CARGO_TOP_PATH, start=1):
        if isinstance(action, tuple):
            node.step(*action)
        else:
            node.step(action)
        if node.levels_completed > base_level:
            print("CUMULATIVE_WIN", index, node.levels_completed)
            return
    print("CUMULATIVE_NO_WIN", avatar_position(node), node.levels_completed)


arena.run_program("dc22", observe)

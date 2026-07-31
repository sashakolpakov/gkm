"""Measure avatar reachability after docking the cargo ring centrally."""
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

import solve
from probe_l6_right import (
    MAIN,
    TOP,
    avatar_position,
    enter_right,
    movement_reach,
)


LEFT = (6, 46, 36)
HUB_TO_BRIDGE = [2, 2, 2, 2, 3, 3, 3, 2, 3]


def observe(env):
    solve.solve(env)
    node = enter_right(env, 3)
    node.step(3)
    node.step(*LEFT)
    for _ in range(3):
        node.step(4)
        node.step(3)
        node.step(*LEFT)
    node.step(4)
    node.step(*MAIN)
    print("DOCKED_HUB", avatar_position(node), node.levels_completed)
    for action in HUB_TO_BRIDGE:
        node.step(action)
    node.step(*TOP)
    node.step(1)
    node.step(3)
    node.step(*TOP)
    print("DOCKED_LEFT", avatar_position(node))
    vertical, vertical_win = movement_reach(node)
    print(
        "DOCKED_VERTICAL",
        sorted(position for position in vertical if position is not None),
        "WIN", vertical_win,
    )
    for _ in range(12):
        node.step(1)
    node.step(*MAIN)
    horizontal, horizontal_win = movement_reach(node)
    print(
        "DOCKED_HORIZONTAL",
        sorted(position for position in horizontal if position is not None),
        "WIN", horizontal_win,
    )


arena.run_program("dc22", observe)

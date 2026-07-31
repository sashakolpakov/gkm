"""Test the cleared black corridor from the horizontal rotator."""
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

import solve
from probe_l6_cargo_top import CARGO_TOP_PATH
from probe_l6_ring_route import HUB_TO_BRIDGE
from probe_l6_right import (
    MAIN,
    TOP,
    avatar_position,
    enter_right,
    movement_reach,
)


def observe(env):
    solve.solve(env)
    node = enter_right(env, 3)
    for action in CARGO_TOP_PATH:
        if isinstance(action, tuple):
            node.step(*action)
        else:
            node.step(action)
    node.step(2)
    node.step(*MAIN)
    for action in HUB_TO_BRIDGE:
        node.step(action)
    node.step(*TOP)
    node.step(1)
    node.step(3)
    node.step(*TOP)
    for _ in range(12):
        node.step(1)
    print("CLEARED_CENTER", avatar_position(node))
    node.step(*MAIN)
    reached, win = movement_reach(node)
    print(
        "CLEARED_HORIZONTAL",
        sorted(position for position in reached if position is not None),
        "VANISHED", None in reached, "WIN", win,
    )


arena.run_program("dc22", observe)

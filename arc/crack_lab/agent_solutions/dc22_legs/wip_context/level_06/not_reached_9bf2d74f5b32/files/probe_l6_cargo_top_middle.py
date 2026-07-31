"""Pair the top-terminal cargo with the upper-middle avatar region."""
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
    base_level = env.levels_completed
    cases = (
        ("MIDDLE_RIGHT", 14, 4),
        ("MIDDLE_LEFT", 17, 4),
        ("TOP", len(CARGO_TOP_PATH), 2),
    )
    for label, prefix, return_to_center in cases:
        node = enter_right(env, 3)
        for action in CARGO_TOP_PATH[:prefix]:
            if isinstance(action, tuple):
                node.step(*action)
            else:
                node.step(action)
        node.step(return_to_center)
        node.step(*MAIN)
        for action in HUB_TO_BRIDGE:
            node.step(action)
        node.step(*TOP)
        node.step(1)
        node.step(3)
        node.step(*TOP)
        won = None
        for step in range(1, 20):
            node.step(1)
            if node.levels_completed > base_level:
                won = step
                break
        reached, win = movement_reach(node)
        print(
            "CARGO_MIDDLE_CASE", label, avatar_position(node),
            won, win, node.levels_completed,
        )


arena.run_program("dc22", observe)

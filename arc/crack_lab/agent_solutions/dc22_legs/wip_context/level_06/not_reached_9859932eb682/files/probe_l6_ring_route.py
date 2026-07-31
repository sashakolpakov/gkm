"""Test physical reachability from the central rotator to the lifted ring."""
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


UP_CONTROL = (6, 50, 34)
HUB_TO_BRIDGE = [2, 2, 2, 2, 3, 3, 3, 2, 3]


def observe(env):
    solve.solve(env)
    node = enter_right(env, 3)
    node.step(1)
    node.step(*UP_CONTROL)
    node.step(2)
    node.step(*MAIN)
    print("BACK_HUB", avatar_position(node), node.levels_completed)
    for action in HUB_TO_BRIDGE:
        node.step(action)
    print("BRIDGE4", avatar_position(node))
    node.step(*TOP)
    node.step(1)
    node.step(3)
    node.step(*TOP)
    print("LEFT_AGAIN", avatar_position(node))
    for _ in range(12):
        node.step(1)
    print("ROTATOR_VERTICAL", avatar_position(node))
    node.step(*MAIN)
    print("ROTATOR_HORIZONTAL", avatar_position(node))
    reached, win = movement_reach(node)
    print(
        "RING_REACH", sorted(
            position for position in reached if position is not None
        ),
        "VANISHED", None in reached, "WIN", win,
    )


if __name__ == "__main__":
    arena.run_program("dc22", observe)

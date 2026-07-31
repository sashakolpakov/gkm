"""Visit the middle/top docks, then return cargo to the central dock."""
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


MOVEMENT = {"U": 1, "D": 2, "L": 3, "R": 4}
INVERSE = {"U": 2, "D": 1, "L": 4, "R": 3}
CONTROL = {
    "U": (6, 50, 34),
    "D": (6, 50, 40),
    "L": (6, 46, 36),
    "R": (6, 54, 36),
}


def command(node, current, target):
    node.step(INVERSE[current])
    node.step(MOVEMENT[target])
    node.step(*CONTROL[target])
    return target


def observe(env):
    solve.solve(env)
    base_level = env.levels_completed
    node = enter_right(env, 3)
    for action in CARGO_TOP_PATH:
        if isinstance(action, tuple):
            node.step(*action)
        else:
            node.step(action)
    current = "U"
    for index, target in enumerate("DDDRRDDLDLLLL", start=1):
        current = command(node, current, target)
        if node.levels_completed > base_level:
            print("CARGO_ROUNDTRIP_WIN", index, target)
            return
        print(
            "CARGO_ROUNDTRIP_STEP", index, target,
            node.levels_completed,
        )
        if index in (12, 13):
            branch = node.clone()
            branch.step(INVERSE[current])
            branch.step(*MAIN)
            hub = avatar_position(branch)
            for _ in range(3):
                branch.step(*SELECTOR)
            branch.step(*MAIN)
            reached, win = movement_reach(branch)
            print(
                "CARGO_ROUNDTRIP_TOP", index, hub,
                avatar_position(branch), win, branch.levels_completed,
            )
    print("CARGO_ROUNDTRIP_NO_WIN", node.levels_completed)


arena.run_program("dc22", observe)

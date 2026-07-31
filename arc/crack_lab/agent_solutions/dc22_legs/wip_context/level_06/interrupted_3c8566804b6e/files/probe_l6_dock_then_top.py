"""Test hidden dock visitation before the cargo's top terminal."""
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

import solve
from probe_l6_cargo_top import CARGO_TOP_PATH
from probe_l6_right import enter_right
from probe_l6_right import MAIN, SELECTOR, avatar_position


LEFT = (6, 46, 36)
RIGHT = (6, 54, 36)


def observe(env):
    solve.solve(env)
    base_level = env.levels_completed
    for depth in (3, 4):
        node = enter_right(env, 3)
        node.step(3)
        for _ in range(depth):
            node.step(*LEFT)
        print("DOCK_VISIT", depth, node.levels_completed - base_level)
        node.step(4)
        node.step(4)
        for _ in range(depth):
            node.step(*RIGHT)
        returned = node.clone()
        returned.step(3)
        returned.step(*MAIN)
        returned.step(*SELECTOR)
        returned.step(*SELECTOR)
        returned.step(*MAIN)
        print(
            "DOCK_RETURN_STATE1", depth,
            avatar_position(returned), returned.levels_completed,
        )
        node.step(3)
        for index, action in enumerate(CARGO_TOP_PATH, start=1):
            if isinstance(action, tuple):
                node.step(*action)
            else:
                node.step(action)
            if node.levels_completed > base_level:
                print("DOCK_THEN_TOP_WIN", depth, index)
                break
        else:
            print("DOCK_THEN_TOP_NO_WIN", depth, node.levels_completed)


arena.run_program("dc22", observe)

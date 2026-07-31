"""Recheck portal destinations after placing the central ring."""
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

import solve
from probe_l6_right import (
    MAIN,
    SELECTOR,
    avatar_position,
    enter_right,
    movement_reach,
    scan_controls,
)


UP_CONTROL = (6, 50, 34)


def observe(env):
    solve.solve(env)
    base_level = env.levels_completed
    lifted = enter_right(env, 3)
    lifted.step(1)
    lifted.step(*UP_CONTROL)
    lifted.step(2)
    lifted.step(*MAIN)
    print("POST_RING_HUB", avatar_position(lifted), lifted.levels_completed)
    for selector_offset in range(4):
        branch = lifted.clone()
        for _ in range(selector_offset):
            branch.step(*SELECTOR)
        branch.step(*MAIN)
        destination = avatar_position(branch)
        print(
            "POST_RING_DEST", selector_offset, destination,
            branch.levels_completed, branch.levels_completed - base_level,
        )
        reached, win = movement_reach(branch)
        print("POST_RING_REACH_WIN", selector_offset, win)
        if destination == (4, 4):
            print("POST_RING_TOP_CONTROLS", scan_controls(branch))


arena.run_program("dc22", observe)

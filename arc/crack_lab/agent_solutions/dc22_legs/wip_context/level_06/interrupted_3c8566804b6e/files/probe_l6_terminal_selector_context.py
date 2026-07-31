"""Test selector state at the exact top-dock arrival transition."""
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


UP = (6, 50, 34)


def observe(env):
    solve.solve(env)
    base_level = env.levels_completed
    preterminal = enter_right(env, 3)
    for action in CARGO_TOP_PATH[:-3]:
        if isinstance(action, tuple):
            preterminal.step(*action)
        else:
            preterminal.step(action)
    for selector_offset in range(4):
        branch = preterminal.clone()
        for _ in range(selector_offset):
            branch.step(*SELECTOR)
        branch.step(*UP)
        print(
            "TERMINAL_SELECTOR_CONTEXT", selector_offset,
            branch.levels_completed,
            branch.levels_completed - base_level,
        )
        branch.step(2)
        for _ in range((-selector_offset) % 4):
            branch.step(*SELECTOR)
        branch.step(*MAIN)
        for _ in range(3):
            branch.step(*SELECTOR)
        branch.step(*MAIN)
        reached, win = movement_reach(branch)
        print(
            "TERMINAL_SELECTOR_TOP", selector_offset,
            avatar_position(branch), win, branch.levels_completed,
        )


arena.run_program("dc22", observe)

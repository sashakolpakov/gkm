"""Consume both upper glyph halves, then map all selector destinations."""
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

import perception
import solve
from probe_l6_right import (
    MAIN,
    SELECTOR,
    TOP,
    TO_REMOTE_PAD,
    avatar_position,
    enter_glyph,
    movement_reach,
)


def clear_left_half(root):
    node = root.clone()
    # The right half was consumed while entering; loop through the left half.
    for action in (1, 3, 2, 1, 4, 2):
        node.step(action)
    return node


def return_to_hub(node):
    for _ in range(13):
        node.step(2)
    for _ in range(6):
        node.step(2)
    node.step(*TOP)
    node.step(*TOP)
    node.step(4)
    node.step(*TOP)
    node.step(2)
    node.step(*TOP)
    for action in TO_REMOTE_PAD:
        node.step(action)


def glyph(frame):
    return perception.arr(frame)[18:20, 6:10].tolist()


def observe(env):
    solve.solve(env)
    base_level = env.levels_completed
    one = enter_glyph(env)
    both = clear_left_half(one)
    print(
        "SECOND_GLYPH", glyph(one.frame()), glyph(both.frame()),
        avatar_position(both), both.levels_completed,
    )
    for selector_steps in range(4):
        branch = both.clone()
        for _ in range(selector_steps):
            branch.step(*SELECTOR)
        return_to_hub(branch)
        hub = avatar_position(branch)
        branch.step(*MAIN)
        destination = avatar_position(branch)
        reached, win = movement_reach(branch)
        print(
            "SECOND_GLYPH_DEST", selector_steps, hub, destination,
            "REACH", sorted(
                position for position in reached if position is not None
            ),
            "WIN", win,
            "LEVEL", branch.levels_completed - base_level,
        )


arena.run_program("dc22", observe)

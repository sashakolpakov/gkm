"""Choose the safe lateral lane after two central catch advances."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from legs import _cell_shape, click_action
from probe_l9_c4_climb import compact, enter_c4_up


def enter_central_top(env):
    enter_c4_up(env)
    env.step(*click_action(5, 4))
    env.step(*click_action(5, 4))


def probe(env):
    enter_central_top(env)
    print("CENTRAL_TOP", compact(env))
    print(
        "SHAPES",
        {
            row: tuple(_cell_shape(env.frame(), row, col) for col in range(8))
            for row in (5, 6)
        },
    )
    right = env.clone()
    right.step(*click_action(6, 5))
    right.step(4)
    print("HANDOFF", 5, compact(right))
    climbed = right.clone()
    climbed.step(*click_action(5, 5))
    print("CLIMB", 5, compact(climbed))

    left = env.clone()
    for target in (3, 2, 1, 0):
        left.step(*click_action(6, target))
        left.step(3)
        print("HANDOFF", target, compact(left))
        climbed = left.clone()
        climbed.step(*click_action(5, target))
        print("CLIMB", target, compact(climbed))


arena.run_program("bp35", probe)

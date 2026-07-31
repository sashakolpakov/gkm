"""Find the minimal supported descent that crosses back left of column 3."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from legs import _cell_shape, click_action
from probe_l9_cross_c6 import compact, switches
from probe_l9_right_climb import enter_far_right


def enter_right_down(env):
    enter_far_right(env)
    env.step(*click_action(5, 7))
    env.step(6, 3, 33)


def probe(env):
    enter_right_down(env)
    env.step(3)
    env.step(3)
    env.step(*click_action(4, 4))
    env.step(3)
    print("MIDDLE", compact(env))
    print(
        "SHAPES",
        {
            row: tuple(_cell_shape(env.frame(), row, col) for col in range(8))
            for row in (4, 5, 6)
        },
    )
    node = env.clone()
    for depth in range(1, 8):
        node.step(*click_action(5, 4))
        print("DROP", depth, compact(node))
        crossed = node.clone()
        crossed.step(*click_action(4, 3))
        crossed.step(3)
        print("CROSS", depth, compact(crossed))
        for switch in switches(crossed):
            flipped = crossed.clone()
            flipped.step(*switch)
            print("FLIP", depth, switch, compact(flipped))


if __name__ == "__main__":
    arena.run_program("bp35", probe)

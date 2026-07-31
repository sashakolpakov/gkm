"""Cross the second upper wall with two supported drops and a flip back."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from legs import click_action
from probe_l9_upper_cross_c3 import compact, enter_upper_ten, switches


def enter_right_landing(env):
    enter_upper_ten(env)
    for col in (3, 4, 5):
        env.step(*click_action(6, col))
        env.step(4)
    env.step(6, 3, 3)


def probe(env):
    enter_right_landing(env)
    print("LANDING", compact(env))
    node = env.clone()
    for depth in range(1, 3):
        node.step(*click_action(5, 5))
        print("DROP", depth, compact(node))
    for col in (6, 7):
        node.step(*click_action(4, col))
        node.step(4)
        print("HANDOFF", col, compact(node))
    for switch in switches(node):
        child = node.clone()
        child.step(*switch)
        print("FLIP", switch, compact(child))


if __name__ == "__main__":
    arena.run_program("bp35", probe)

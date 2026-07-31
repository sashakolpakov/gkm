"""Trace the unique post-switch column-4 central ascent."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from legs import click_action
from probe_l9_cross_c6 import compact
from probe_l9_return_left import enter_right_down


def enter_c4_up(env):
    enter_right_down(env)
    env.step(3)
    env.step(3)
    env.step(*click_action(4, 4))
    env.step(3)
    env.step(6, 3, 41)


def probe(env):
    enter_c4_up(env)
    node = env.clone()
    print("C4_UP", compact(node))
    for advance in range(1, 11):
        if node.terminal():
            break
        node.step(*click_action(5, 4))
        print("CLIMB", advance, compact(node))


if __name__ == "__main__":
    arena.run_program("bp35", probe)

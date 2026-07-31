"""Climb the far-right catch through the second horizontal barrier."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from legs import click_action
from probe_l9_cross_c6 import compact, enter_right_landing, switches


def enter_far_right(env):
    enter_right_landing(env)
    env.step(*click_action(5, 5))
    env.step(*click_action(5, 5))
    for col in (6, 7):
        env.step(*click_action(4, col))
        env.step(4)
    env.step(6, 3, 5)


def probe(env):
    enter_far_right(env)
    node = env.clone()
    print("FAR_RIGHT", compact(node))
    for advance in range(1, 13):
        if node.terminal():
            break
        node.step(*click_action(5, 7))
        print("CLIMB", advance, compact(node))


if __name__ == "__main__":
    arena.run_program("bp35", probe)

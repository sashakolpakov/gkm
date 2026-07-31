"""Flip upward from each supported column in the right-down landing."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from legs import click_action
from probe_l9_cross_c6 import compact, switches
from probe_l9_return_left import enter_right_down


def probe(env):
    enter_right_down(env)
    node = env.clone()
    for col in (7, 6, 5):
        if col < 7:
            node.step(3)
        print("COLUMN", col, compact(node))
        for switch in switches(node):
            child = node.clone()
            child.step(*switch)
            print("FLIP", col, switch, compact(child))
            climbed = child.clone()
            climbed.step(*click_action(5, col))
            print("CLIMB", col, compact(climbed))
    node.step(*click_action(4, 4))
    node.step(3)
    print("COLUMN", 4, compact(node))
    for switch in switches(node):
        child = node.clone()
        child.step(*switch)
        print("FLIP", 4, switch, compact(child))
        climbed = child.clone()
        climbed.step(*click_action(5, 4))
        print("CLIMB", 4, compact(climbed))


arena.run_program("bp35", probe)

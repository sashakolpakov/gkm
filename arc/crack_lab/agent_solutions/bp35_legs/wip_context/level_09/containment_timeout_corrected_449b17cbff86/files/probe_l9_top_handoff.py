"""Escape the lethal right shaft and choose the next upper lane."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from legs import click_action
from probe_l9_cross_c6 import compact
from probe_l9_right_climb import enter_far_right


def enter_top(env):
    enter_far_right(env)
    for _ in range(7):
        env.step(*click_action(5, 7))


def probe(env):
    enter_top(env)
    root = env.clone()
    print("TOP", compact(root))
    cleared = root.clone()
    cleared.step(*click_action(3, 2))
    climbed = cleared.clone()
    climbed.step(*click_action(5, 7))
    print("CLEAR_THEN_CLIMB", compact(climbed))

    node = cleared
    for target in range(6, -1, -1):
        node.step(*click_action(6, target))
        node.step(3)
        print("HANDOFF", target, compact(node))
        child = node.clone()
        child.step(*click_action(5, target))
        print("CLIMB", target, compact(child))


if __name__ == "__main__":
    arena.run_program("bp35", probe)

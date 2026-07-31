"""Map downward landings from each far-right ladder height."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from legs import click_action
from probe_l9_cross_c6 import compact, switches
from probe_l9_right_climb import enter_far_right


def probe(env):
    enter_far_right(env)
    node = env.clone()
    for height in range(1, 7):
        node.step(*click_action(5, 7))
        print("HEIGHT", height, compact(node))
        for switch in switches(node):
            child = node.clone()
            child.step(*switch)
            print("FLIP", height, switch, compact(child))


arena.run_program("bp35", probe)

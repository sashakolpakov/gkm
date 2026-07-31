"""Carry the staged bridge right at the verified below-wall depth."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from legs import click_action
from probe_l9_short_landing import compact, enter_short_landing, switches


def probe(env):
    enter_short_landing(env)
    root = env.clone()
    root.step(*click_action(5, 2))
    root.step(*click_action(5, 2))
    print("DEPTH2", compact(root))
    node = root.clone()
    for target in range(3, 8):
        node.step(*click_action(4, target))
        node.step(4)
        print("CROSS", target, compact(node))
        for switch in switches(node):
            flipped = node.clone()
            flipped.step(*switch)
            print("FLIP", target, switch, compact(flipped))


arena.run_program("bp35", probe)

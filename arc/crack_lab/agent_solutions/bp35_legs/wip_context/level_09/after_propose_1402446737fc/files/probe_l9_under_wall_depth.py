"""Find the first staged drop deep enough to cross the upper overhang."""

from probe_l9_short_landing import (
    compact,
    enter_short_landing,
    switches,
)

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from legs import click_action


def probe(env):
    enter_short_landing(env)
    node = env.clone()
    for depth in range(1, 7):
        node.step(*click_action(5, 2))
        print("DROP", depth, compact(node))
        crossed = node.clone()
        crossed.step(*click_action(4, 3))
        crossed.step(4)
        print("HANDOFF", depth, compact(crossed))
        for switch in switches(crossed):
            flipped = crossed.clone()
            flipped.step(*switch)
            print("FLIP", depth, switch, compact(flipped))


arena.run_program("bp35", probe)

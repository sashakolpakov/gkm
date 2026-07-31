"""Test controlled falls through each small hazard in the right-down room."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from legs import _cell_shape, click_action
from probe_l9_cross_c6 import compact
from probe_l9_return_left import enter_right_down


def probe(env):
    enter_right_down(env)
    node = env.clone()
    for col in (7, 6, 5):
        if col < 7:
            node.step(3)
        print(
            "COLUMN",
            col,
            {"below": _cell_shape(node.frame(), 5, col), "state": compact(node)},
        )
        child = node.clone()
        child.step(*click_action(5, col))
        print("FALL", col, compact(child))


arena.run_program("bp35", probe)

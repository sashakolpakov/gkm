"""Flip during the direct boosted descent while two controls remain."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from probe_l9_boosted_gate_alignment import gate
from probe_l9_boosted_supported_landing import report
from probe_l9_control_row import controls
from probe_l9_route_deletions import enter_level_9


def direct(root, depth):
    child = gate(root, 1)
    child.step(4)
    for _ in range(depth):
        child.step(6, 27, 33)
    return child


def probe(env):
    enter_level_9(env)
    for depth in range(3):
        root = direct(env, depth)
        report((depth, "PRE"), root)
        for switch_index in range(len(controls(root))):
            child = root.clone()
            switch = controls(child)[switch_index]
            child.step(*switch)
            report((depth, switch_index, switch), child)
            if child.terminal():
                continue
            for action in (3, 4):
                moved = child.clone()
                moved.step(action)
                report((depth, switch_index, action), moved)


if __name__ == "__main__":
    arena.run_program("bp35", probe)

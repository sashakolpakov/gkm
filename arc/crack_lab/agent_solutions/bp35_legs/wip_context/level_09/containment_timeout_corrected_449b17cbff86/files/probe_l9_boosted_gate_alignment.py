"""Align the four-control frontier before its second gravity reversal."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from probe_l9_boosted_supported_landing import report
from probe_l9_control_row import controls
from probe_l9_preserve_boost import boosted
from probe_l9_route_deletions import enter_level_9


def gate(root, lefts):
    child = boosted(root)
    child.step(*controls(child)[0])
    child.step(6, 21, 39)
    for _ in range(lefts):
        child.step(3)
    return child


def probe(env):
    enter_level_9(env)
    for lefts in range(3):
        root = gate(env, lefts)
        report(("PRE", lefts), root)
        for switch_index in range(len(controls(root))):
            child = root.clone()
            switch = controls(child)[switch_index]
            child.step(*switch)
            report(("FLIP", lefts, switch_index, switch), child)
            if child.terminal():
                continue
            child.step(4)
            report(("RIGHT", lefts, switch_index), child)


if __name__ == "__main__":
    arena.run_program("bp35", probe)

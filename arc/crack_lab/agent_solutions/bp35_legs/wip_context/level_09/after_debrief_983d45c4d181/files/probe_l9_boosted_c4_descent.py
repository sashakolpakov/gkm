"""Use the propagated catch to support the boosted column-four descent."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from probe_l9_boosted_gate_alignment import gate
from probe_l9_boosted_supported_landing import report
from probe_l9_control_row import controls
from probe_l9_route_deletions import enter_level_9


def c4(root):
    child = gate(root, 1)
    child.step(6, 21, 21)
    child.step(*controls(child)[0])
    child.step(6, 27, 39)
    child.step(4)
    return child


def probe(env):
    enter_level_9(env)
    child = c4(env)
    report("C4", child)
    for depth in range(1, 10):
        child.step(6, 27, 33)
        report(("DROP", depth), child)
        if child.terminal() or int(child.levels_completed) >= 9:
            break


if __name__ == "__main__":
    arena.run_program("bp35", probe)

"""Stage one catch before the boosted route's second gravity reversal."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from probe_l9_boosted_gate_alignment import gate
from probe_l9_boosted_supported_landing import report
from probe_l9_control_row import controls
from probe_l9_route_deletions import enter_level_9


def probe(env):
    enter_level_9(env)
    targets = (
        (21, 21),
        (21, 39),
        (27, 39),
        (21, 45),
        (27, 45),
        (21, 51),
        (27, 51),
        (21, 57),
        (27, 57),
        (33, 57),
    )
    for target in targets:
        child = gate(env, 1)
        child.step(6, *target)
        if child.terminal() or not controls(child):
            report((target, "STAGED"), child)
            continue
        switch = controls(child)[0]
        child.step(*switch)
        report((target, "FLIP", switch), child)
        if child.terminal():
            continue
        child.step(4)
        child.step(6, 27, 33)
        report((target, "C4_DROP"), child)


if __name__ == "__main__":
    arena.run_program("bp35", probe)

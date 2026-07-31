"""Descend from the boosted gate without spending the redundant second flip."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from probe_l9_boosted_gate_alignment import gate
from probe_l9_boosted_supported_landing import report
from probe_l9_route_deletions import enter_level_9


def run(root, name, staged):
    child = gate(root, 1)
    if staged:
        child.step(6, 21, 21)
    child.step(4)
    report((name, "C4"), child)
    for depth in range(1, 10):
        child.step(6, 27, 33)
        report((name, depth), child)
        if child.terminal() or int(child.levels_completed) >= 9:
            break


def probe(env):
    enter_level_9(env)
    run(env, "PLAIN", False)
    run(env, "STAGED", True)


if __name__ == "__main__":
    arena.run_program("bp35", probe)

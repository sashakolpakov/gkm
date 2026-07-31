"""Screen controls at the resurfaced compressed right-shaft landing."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from probe_l9_boosted_supported_landing import report
from probe_l9_control_row import controls
from probe_l9_fast_right_handoff import reversed_root
from probe_l9_right_trap_stage import full_catches
from probe_l9_route_deletions import enter_level_9


def resurfaced(root):
    child = reversed_root(root)
    for _ in range(5):
        child.step(6, 27, 33)
    return child


def probe(env):
    enter_level_9(env)
    child = resurfaced(env)
    report(("PRE", sys.argv[1]), child)
    print("FULL", full_catches(child), flush=True)
    switch = controls(child)[int(sys.argv[1])]
    child.step(*switch)
    report(("FLIP", switch), child)
    if child.terminal():
        return
    for action in ((6, 27, 33), 3, 4, (6, 27, 27), (6, 21, 39)):
        branch = child.clone()
        branch.step(*action) if isinstance(action, tuple) else branch.step(action)
        report(("ACTION", action), branch)


if __name__ == "__main__":
    arena.run_program("bp35", probe)

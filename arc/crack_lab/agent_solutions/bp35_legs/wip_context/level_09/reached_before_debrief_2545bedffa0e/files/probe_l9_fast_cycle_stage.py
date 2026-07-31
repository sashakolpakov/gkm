"""Stage a column-four landing before the resurfaced gravity reversal."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from probe_l9_boosted_supported_landing import report
from probe_l9_control_row import controls
from probe_l9_fast_cycle_switch import resurfaced
from probe_l9_route_deletions import enter_level_9


def staged(root, y):
    child = resurfaced(root)
    child.step(6, 21, y)
    return child


def probe(env):
    enter_level_9(env)
    y = int(sys.argv[1])
    switch_index = int(sys.argv[2])
    child = staged(env, y)
    report((y, switch_index, "STAGED"), child)
    visible = controls(child)
    child.step(*visible[switch_index])
    report((y, switch_index, "FLIP"), child)
    if child.terminal():
        return
    for action in ((6, 27, 33), (6, 27, 45), 3, 4):
        branch = child.clone()
        branch.step(*action) if isinstance(action, tuple) else branch.step(action)
        report((y, switch_index, action), branch)


if __name__ == "__main__":
    arena.run_program("bp35", probe)

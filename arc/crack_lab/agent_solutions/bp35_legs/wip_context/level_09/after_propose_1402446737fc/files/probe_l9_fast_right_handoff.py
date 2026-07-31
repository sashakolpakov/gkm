"""Test handoffs after the compressed supported right-shaft reversal."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from probe_l9_boosted_supported_landing import report
from probe_l9_control_row import controls
from probe_l9_fast_right_support import root_at_right
from probe_l9_right_trap_handoff import BRANCHES
from probe_l9_route_deletions import enter_level_9


def reversed_root(root):
    child = root_at_right(root)
    child.step(6, 21, 33)
    child.step(6, 21, 35)
    for _ in range(3):
        child.step(6, 21, 33)
    child.step(6, 27, 27)
    child.step(4)
    child.step(*controls(child)[-1])
    return child


def probe(env):
    enter_level_9(env)
    child = reversed_root(env)
    report((sys.argv[1], "FLIP"), child)
    if sys.argv[1] == "climb":
        actions = ((6, 27, 33),) * 5
    else:
        child.step(6, 27, 33)
        report((sys.argv[1], 0), child)
        actions = BRANCHES[sys.argv[1]]
    for index, action in enumerate(actions, 1):
        child.step(*action) if isinstance(action, tuple) else child.step(action)
        report((sys.argv[1], index, action), child)
        if child.terminal() or int(child.levels_completed) >= 9:
            break


if __name__ == "__main__":
    arena.run_program("bp35", probe)

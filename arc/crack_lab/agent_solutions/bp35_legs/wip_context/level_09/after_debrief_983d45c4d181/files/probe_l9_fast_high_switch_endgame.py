"""Carry the row-62 switch through the compressed boosted handoff."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from probe_l9_control_row import controls
from probe_l9_fast_boost_endgame import report
from probe_l9_fast_boost_switches import boosted_flip
from probe_l9_route_deletions import enter_level_9


def turned(root, second_index=0):
    child = boosted_flip(root)
    child.step(*controls(child)[0])
    for action in ((6, 21, 39), 3, (6, 15, 39), 3):
        child.step(*action) if isinstance(action, tuple) else child.step(action)
    report((second_index, "PRE_SECOND"), child)
    child.step(*controls(child)[second_index])
    return child


def run(root, name, actions, second_index=0):
    child = turned(root, second_index)
    report((name, 0), child)
    for index, action in enumerate(actions, 1):
        child.step(*action) if isinstance(action, tuple) else child.step(action)
        report((name, index, action), child)
        if child.terminal() or int(child.levels_completed) >= 9:
            break


def probe(env):
    enter_level_9(env)
    variants = {
        "RIGHT": (4,) * 12,
        "LOWER_RIGHT": ((6, 15, 45), 4, 4, 4, 4, 4),
        "ROW_CLEAR_RIGHT": ((6, 21, 39), 4, 4, 4, 4),
        "DROP_C2": ((6, 15, 45),) * 8,
    }
    for name, actions in variants.items():
        run(env, name, actions)
    run(env, "SECOND_LOW", (4,) * 8, second_index=-1)


if __name__ == "__main__":
    arena.run_program("bp35", probe)

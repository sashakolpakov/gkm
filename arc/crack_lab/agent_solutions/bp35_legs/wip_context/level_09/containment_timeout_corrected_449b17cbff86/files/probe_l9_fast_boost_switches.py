"""Compare all switches after the fast route reveals the boosted fourth control."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from probe_l9_control_row import compact, controls
from probe_l9_fast_boost_endgame import FAST_SKIPS, report
from probe_l9_route_deletions import enter_level_9, replay, route


def boosted_flip(root):
    child = replay(root, route(), skips=FAST_SKIPS)
    for action in (
        (6, 21, 39),
        4,
        (6, 27, 39),
        4,
        (6, 27, 33),
        (6, 27, 33),
        (6, 27, 33),
    ):
        child.step(*action) if isinstance(action, tuple) else child.step(action)
    child.step(*controls(child)[0])
    return child


def probe(env):
    enter_level_9(env)
    root = boosted_flip(env)
    report("ROOT", root)
    for switch_index in range(3):
        child = root.clone()
        visible = controls(child)
        child.step(*visible[switch_index])
        report((switch_index, visible[switch_index], "FLIP"), child)
        for action in ((6, 21, 39), 3, (6, 15, 39), 3):
            child.step(*action) if isinstance(action, tuple) else child.step(action)
        report((switch_index, "LEFT_GATE"), child)


if __name__ == "__main__":
    arena.run_program("bp35", probe)

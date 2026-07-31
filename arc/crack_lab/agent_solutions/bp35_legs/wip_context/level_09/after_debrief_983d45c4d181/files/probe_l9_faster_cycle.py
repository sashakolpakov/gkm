"""Graft the verified seven-action prefix compression onto the right-shaft cycle."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from probe_l9_boosted_supported_landing import report
from probe_l9_control_row import controls
from probe_l9_fast_boost_endgame import FAST_SKIPS
from probe_l9_route_deletions import enter_level_9, replay, route


def fast_boosted(root):
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
    return child


def reversed_root(root):
    child = fast_boosted(root)
    child.step(*controls(child)[0])
    child.step(3)
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
    report("FLIP", child)
    for height in range(1, 15):
        child.step(6, 27, 33)
        report(("CLIMB", height), child)
        if child.terminal() or int(child.levels_completed) >= 9:
            break


if __name__ == "__main__":
    arena.run_program("bp35", probe)

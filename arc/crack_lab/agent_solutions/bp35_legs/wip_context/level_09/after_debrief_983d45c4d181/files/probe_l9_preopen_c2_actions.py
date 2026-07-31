"""Probe the propagated catch below the pre-opened column-two trapdoor."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from probe_l9_boosted_supported_landing import report
from probe_l9_preopen_gap_cross import root_state
from probe_l9_route_deletions import enter_level_9


def c2(root):
    child = root_state(root)
    for action in (
        3,
        (6, 21, 39),
        3,
        (6, 15, 39),
        3,
    ):
        child.step(*action) if isinstance(action, tuple) else child.step(action)
    return child


def run(root, name, actions):
    child = c2(root)
    report((name, 0), child)
    for index, action in enumerate(actions, 1):
        child.step(*action) if isinstance(action, tuple) else child.step(action)
        report((name, index, action), child)
        if child.terminal() or int(child.levels_completed) >= 9:
            break


def probe(env):
    enter_level_9(env)
    variants = {
        "BELOW": ((6, 15, 45),) * 10,
        "BELOW_RIGHT": ((6, 15, 45), 4, 4),
        "BELOW_LEFT": ((6, 15, 45), 3, 3),
        "DEEP": (
            (6, 15, 45),
            (6, 15, 51),
            (6, 15, 57),
        ),
    }
    for name, actions in variants.items():
        run(env, name, actions)


if __name__ == "__main__":
    arena.run_program("bp35", probe)

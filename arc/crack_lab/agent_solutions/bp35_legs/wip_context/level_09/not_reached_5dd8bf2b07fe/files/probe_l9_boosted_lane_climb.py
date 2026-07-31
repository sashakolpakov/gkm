"""Attempt the fifth-control climb from lanes adjacent to column four."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from probe_l9_boosted_supported_landing import report
from probe_l9_preserve_boost import boosted
from probe_l9_route_deletions import enter_level_9


def run(root, name, actions):
    child = boosted(root)
    report((name, 0), child)
    for index, action in enumerate(actions, 1):
        child.step(*action) if isinstance(action, tuple) else child.step(action)
        report((name, index, action), child)
        if child.terminal() or int(child.levels_completed) >= 9:
            break


def probe(env):
    enter_level_9(env)
    variants = {
        "LEFT": ((6, 21, 39), 3) + ((6, 21, 33),) * 4,
        "RIGHT": ((6, 33, 39), 4) + ((6, 33, 33),) * 4,
        "FAR_LEFT": (
            (6, 21, 39),
            3,
            (6, 15, 39),
            3,
        )
        + ((6, 15, 33),) * 4,
    }
    for name, actions in variants.items():
        run(env, name, actions)


if __name__ == "__main__":
    arena.run_program("bp35", probe)

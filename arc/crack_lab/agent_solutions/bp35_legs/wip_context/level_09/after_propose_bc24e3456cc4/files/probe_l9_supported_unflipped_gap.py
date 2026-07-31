"""Descend through the lower gap before consuming the retained switch."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from probe_l9_route_deletions import enter_level_9
from probe_l9_skip4_switch_choices import report
from probe_l9_supported_final_alignment import aligned


def run(root, name, actions):
    child = aligned(root, 6, 6)
    report((name, 0), child)
    for index, action in enumerate(actions, 1):
        child.step(*action) if isinstance(action, tuple) else child.step(action)
        report((name, index, action), child)
        if child.terminal() or int(child.levels_completed) >= 9:
            break


def probe(env):
    enter_level_9(env)
    variants = {
        "DROP_C6": ((6, 39, 33),) * 12,
        "RIGHT": (4,) * 8,
        "LEFT": (3,) * 8,
        "DEEP": ((6, 39, 39), (6, 39, 45), (6, 39, 51), (6, 39, 57)),
    }
    for name, actions in variants.items():
        run(env, name, actions)


if __name__ == "__main__":
    arena.run_program("bp35", probe)

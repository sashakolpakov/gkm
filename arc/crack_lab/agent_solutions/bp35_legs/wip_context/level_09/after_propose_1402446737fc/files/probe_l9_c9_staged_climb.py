"""Stage safe supports before opening the lethal above catch at column nine."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from probe_l9_route_deletions import enter_level_9
from probe_l9_skip4_switch_choices import report
from probe_l9_supported_c9_actions import c9


ABOVE = (6, 57, 33)
BELOW = (6, 57, 45)
UPPER_LEFT = (6, 51, 33)
LOWER_LEFT = (6, 51, 45)


def run(root, name, actions):
    child = c9(root)
    report((name, 0), child)
    for index, action in enumerate(actions, 1):
        child.step(*action) if isinstance(action, tuple) else child.step(action)
        report((name, index, action), child)
        if child.terminal() or int(child.levels_completed) >= 9:
            break


def probe(env):
    enter_level_9(env)
    variants = {
        "BELOW_ABOVE": (BELOW, ABOVE),
        "UPPER_LEFT_ABOVE": (UPPER_LEFT, ABOVE),
        "BELOW_UPPER_ABOVE": (BELOW, UPPER_LEFT, ABOVE),
        "LOWER_LEFT_ABOVE": (LOWER_LEFT, ABOVE),
        "BELOW_LEFT_CLIMB": (BELOW, 3, (6, 51, 33)),
        "UPPER_LEFT_LEFT_CLIMB": (UPPER_LEFT, 3, (6, 51, 33)),
    }
    for name, actions in variants.items():
        run(env, name, actions)


if __name__ == "__main__":
    arena.run_program("bp35", probe)

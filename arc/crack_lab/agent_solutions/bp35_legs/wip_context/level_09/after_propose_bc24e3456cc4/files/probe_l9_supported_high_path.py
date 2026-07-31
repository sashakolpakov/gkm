"""Follow the corrected high-high switch path from the arrested landing."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from probe_l9_control_row import controls
from probe_l9_route_deletions import enter_level_9
from probe_l9_skip4_switch_choices import report
from probe_l9_supported_search import supported


def high_high(root):
    child = supported(root)
    child.step(*controls(child)[0])
    child.step(6, 21, 39)
    child.step(3)
    child.step(3)
    report("PRE_SECOND", child)
    child.step(*controls(child)[0])
    return child


def run(root, name, actions):
    child = high_high(root)
    report((name, 0), child)
    for index, action in enumerate(actions, 1):
        child.step(*action) if isinstance(action, tuple) else child.step(action)
        report((name, index, action), child)
        if child.terminal() or int(child.levels_completed) >= 9:
            break


def probe(env):
    enter_level_9(env)
    variants = {
        "RIGHT": (4,) * 10,
        "DROP_C4": (4, 4) + ((6, 27, 33),) * 8,
        "LOWER_C2": ((6, 15, 45), 4, 4, 4, 4),
    }
    for name, actions in variants.items():
        run(env, name, actions)


if __name__ == "__main__":
    arena.run_program("bp35", probe)

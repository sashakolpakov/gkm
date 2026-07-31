"""Open the catch east of the lower-gap landing and trace its exit corridor."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from probe_l9_route_deletions import enter_level_9
from probe_l9_skip4_switch_choices import report
from probe_l9_supported_gap_climb import flipped


def run(root, name, actions):
    child = flipped(root, 6)
    report((name, 0), child)
    for index, action in enumerate(actions, 1):
        child.step(*action) if isinstance(action, tuple) else child.step(action)
        report((name, index, action), child)
        if child.terminal() or int(child.levels_completed) >= 9:
            break


def probe(env):
    enter_level_9(env)
    variants = {
        "CLEAR_WALK": ((6, 45, 39), 4, 4, 4, 4),
        "SUPPORT_WALK": ((6, 45, 45), (6, 45, 39), 4, 4, 4),
        "CLEAR_C8": ((6, 45, 39), 4, (6, 51, 39), 4, 4),
        "CLEAR_BELOW": ((6, 45, 39), 4, (6, 45, 45), 4, 4),
        "FULL_CROSS": (
            (6, 45, 39),
            4,
            (6, 51, 39),
            4,
            (6, 57, 39),
            4,
            4,
            (6, 57, 33),
        ),
    }
    for name, actions in variants.items():
        run(env, name, actions)


if __name__ == "__main__":
    arena.run_program("bp35", probe)

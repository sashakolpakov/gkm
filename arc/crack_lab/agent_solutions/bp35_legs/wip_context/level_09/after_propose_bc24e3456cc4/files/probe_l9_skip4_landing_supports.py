"""Stage the two differing landing catches before the height-four shortcut flip."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from probe_l9_control_row import controls
from probe_l9_route_deletions import enter_level_9
from probe_l9_skip4_switch_choices import before_flip, report


UP = (6, 27, 33)


def staged(root):
    child = before_flip(root)
    for _ in range(4):
        child.step(*UP)
    return child


def run(root, name, actions):
    child = staged(root)
    for action in actions:
        child.step(*action)
    report((name, "STAGED"), child)
    visible = controls(child)
    if not visible or child.terminal():
        return
    child.step(*visible[0])
    report((name, "FLIP"), child)


def probe(env):
    enter_level_9(env)
    variants = {
        "NONE": (),
        "R9C3": ((6, 21, 57),),
        "R8C3": ((6, 21, 51),),
        "R7C3": ((6, 21, 45),),
        "R8C2": ((6, 15, 51),),
        "R9_R8": ((6, 21, 57), (6, 21, 51)),
        "R8_R9": ((6, 21, 51), (6, 21, 57)),
        "R9_C2": ((6, 21, 57), (6, 15, 51)),
    }
    for name, actions in variants.items():
        run(env, name, actions)


if __name__ == "__main__":
    arena.run_program("bp35", probe)

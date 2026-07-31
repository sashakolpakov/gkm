"""Descend and enter exterior c9 from the 27-action four-control shortcut."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from probe_l9_boosted_supported_landing import report
from probe_l9_control_row import controls
from probe_l9_early21_right import early_right, state
from probe_l9_route_deletions import enter_level_9


def flipped(root):
    child = early_right(root)
    child.step(*controls(child)[0])
    return child


def landing(root, depth=2):
    child = flipped(root)
    for _ in range(depth):
        child.step(6, 45, 33)
    return child


def probe(env):
    enter_level_9(env)
    child = flipped(env)
    state("FLIPPED", child)
    for depth in range(1, 5):
        child.step(6, 45, 33)
        state(("DROP", depth), child)
        if child.terminal():
            break
    for depth in (1, 2):
        branch = landing(env, depth)
        for label, action in (
            ("C8_CLEAR", (6, 51, 27)),
            ("C8_MOVE", (4,)),
            ("C9_CLEAR", (6, 57, 29)),
            ("C9_MOVE", (4,)),
        ):
            branch.step(*action)
            state((depth, label), branch)
            if branch.terminal():
                break


if __name__ == "__main__":
    arena.run_program("bp35", probe)

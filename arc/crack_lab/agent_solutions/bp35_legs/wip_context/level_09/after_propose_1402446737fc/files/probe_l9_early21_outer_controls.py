"""Flip the nine controls revealed after bypassing the solid wall through c0."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from probe_l9_control_row import controls
from probe_l9_early21_no_control_height6 import early_height6
from probe_l9_early21_right import state
from probe_l9_route_deletions import enter_level_9


LEFT_EXIT = (
    (6, 27, 39), 3,
    (6, 21, 39), 3,
    (6, 15, 39), 3,
    (6, 9, 39), 3,
    3,
)


def outer_left(root):
    child = early_height6(root)
    for action in LEFT_EXIT:
        child.step(*action) if isinstance(action, tuple) else child.step(action)
    return child


def probe(env):
    enter_level_9(env)
    root = outer_left(env)
    state("OUTER_LEFT", root)
    winner = root.clone()
    winner.step(*controls(winner)[0])
    winner.step(4)
    winner.step(4)
    state("WINNER", winner)
    visible = controls(root)
    for switch_index in (0, len(visible) // 2, len(visible) - 1):
        child = root.clone()
        switch = controls(child)[switch_index]
        child.step(*switch)
        state((switch_index, switch, "DOWN"), child)
        if child.terminal():
            continue
        for action in ((3,), (4,), (6, 3, 45), (6, 3, 33)):
            branch = child.clone()
            branch.step(*action)
            state((switch_index, "ACTION", action), branch)


if __name__ == "__main__":
    arena.run_program("bp35", probe)

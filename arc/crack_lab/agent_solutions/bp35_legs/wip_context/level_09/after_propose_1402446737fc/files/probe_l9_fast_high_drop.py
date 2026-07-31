"""Flip the retained low switch at each depth of the column-four descent."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from probe_l9_control_row import controls
from probe_l9_fast_boost_endgame import report
from probe_l9_fast_high_switch_endgame import turned
from probe_l9_route_deletions import enter_level_9


DROP = (6, 27, 33)


def c4(root):
    child = turned(root)
    child.step(4)
    child.step(4)
    return child


def probe(env):
    enter_level_9(env)
    for depth in range(7):
        child = c4(env)
        for _ in range(depth):
            child.step(*DROP)
        report((depth, "PRE"), child)
        visible = controls(child)
        if not visible or child.terminal():
            continue
        child.step(*visible[-1])
        report((depth, "FLIP"), child)
        left = child.clone()
        right = child
        for _ in range(6):
            left.step(3)
            if left.terminal() or int(left.levels_completed) >= 9:
                break
        report((depth, "LEFT6"), left)
        for _ in range(6):
            right.step(4)
            if right.terminal() or int(right.levels_completed) >= 9:
                break
        report((depth, "RIGHT6"), right)


if __name__ == "__main__":
    arena.run_program("bp35", probe)

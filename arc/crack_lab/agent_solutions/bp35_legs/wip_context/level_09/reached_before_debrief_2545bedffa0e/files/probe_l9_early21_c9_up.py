"""Preserve an upper control through the shortcut c9 climb for a wall reversal."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from probe_l9_control_row import controls
from probe_l9_early21_c9 import landing
from probe_l9_early21_right import state
from probe_l9_route_deletions import enter_level_9


def c9(root):
    child = landing(root, 2)
    for action in ((6, 51, 27), 4, (6, 57, 29), 4):
        child.step(*action) if isinstance(action, tuple) else child.step(action)
    return child


def climbed(root, switch_index):
    child = c9(root)
    child.step(*controls(child)[switch_index])
    for _ in range(7):
        child.step(6, 57, 33)
    return child


def probe(env):
    enter_level_9(env)
    root = c9(env)
    state("C9", root)
    for switch_index in range(len(controls(root))):
        child = root.clone()
        switch = controls(child)[switch_index]
        child.step(*switch)
        state((switch_index, switch, "UP"), child)
        for height in range(1, 8):
            child.step(6, 57, 33)
            state((switch_index, "HEIGHT", height), child)
            if child.terminal():
                break
        if not child.terminal() and controls(child):
            remaining = controls(child)[0]
            child.step(*remaining)
            state((switch_index, remaining, "WALL_FLIP"), child)


if __name__ == "__main__":
    arena.run_program("bp35", probe)

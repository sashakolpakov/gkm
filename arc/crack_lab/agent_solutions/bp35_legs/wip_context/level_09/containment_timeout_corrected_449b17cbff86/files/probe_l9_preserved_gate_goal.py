"""Carry the gate-preserved switch through the aligned trapdoor to the goal wall."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from probe_l9_boosted_supported_landing import report
from probe_l9_control_row import controls
from probe_l9_multi_skip_bottom import root_for
from probe_l9_oneflip_trapdoors import yellows
from probe_l9_route_deletions import enter_level_9


def step(env, action):
    env.step(*action) if isinstance(action, tuple) else env.step(action)


def preserved_gate(root):
    child = root_for(root, 3)
    for action in ((6, 21, 39), 4, (6, 27, 39), 4):
        step(child, action)
    return child


def probe(env):
    enter_level_9(env)
    root = preserved_gate(env)
    root.step(6, 27, 33)
    report("DROP", root)
    for first_index in range(len(controls(root))):
        child = root.clone()
        first = controls(child)[first_index]
        child.step(*first)
        child.step(3)
        child.step(3)
        report((first_index, first, "ALIGNED"), child)
        visible = controls(child)
        if len(visible) < 2 or child.terminal():
            continue
        left_yellow = min(yellows(child), key=lambda action: action[1])
        child.step(*left_yellow)
        report((first_index, "OPENED"), child)
        visible = controls(child)
        if not visible:
            continue
        child.step(*visible[-1])
        report((first_index, "DEEP"), child)
        if child.terminal():
            continue
        for index in range(1, 4):
            child.step(4)
            report((first_index, "RIGHT", index), child)
        visible = controls(child)
        if visible:
            child.step(*visible[-1])
            report((first_index, "WALL_FLIP"), child)


if __name__ == "__main__":
    arena.run_program("bp35", probe)

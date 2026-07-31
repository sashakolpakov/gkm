"""Shortcut directly from the 21-action opening climb to the c7 frontier."""

import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from probe_l9_boosted_supported_landing import report
from probe_l9_control_row import controls
from probe_l9_handoff_clickmap import avatar
from probe_l9_presecond_right_flip import right_end
from probe_l9_right_trap_stage import full_catches
from probe_l9_route_deletions import enter_level_9, route, step


def early21(root):
    child = root.clone()
    for _, action in route()[:21]:
        step(child, action)
    return child


def early_right(root):
    child = early21(root)
    for x in (33, 39, 45):
        child.step(6, x, 39)
        child.step(4)
    return child


def physical(env):
    frame = np.asarray(env.frame()).copy()
    frame[63, :] = 0
    return frame


def state(label, child):
    report(label, child)
    print(
        "STATE",
        label,
        "terminal",
        bool(child.terminal()),
        "avatar",
        avatar(child),
        "controls",
        controls(child),
        "full",
        full_catches(child),
        flush=True,
    )


def probe(env):
    enter_level_9(env)
    child = early_right(env)
    target = right_end(env)
    state("EARLY_RIGHT", child)
    print(
        "COMPARE",
        "pixels",
        int(np.count_nonzero(physical(child) != physical(target))),
        "target_controls",
        controls(target),
        flush=True,
    )
    for switch_index, switch in enumerate(controls(child)):
        branch = child.clone()
        branch.step(*switch)
        state(("FLIP", switch_index, switch), branch)


if __name__ == "__main__":
    arena.run_program("bp35", probe)

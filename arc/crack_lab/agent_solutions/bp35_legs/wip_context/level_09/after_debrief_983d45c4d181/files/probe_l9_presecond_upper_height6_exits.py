"""Traverse both lateral exits from the six-rise c5 solid-ceiling frontier."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from probe_l9_boosted_supported_landing import report
from probe_l9_control_row import controls
from probe_l9_handoff_clickmap import avatar
from probe_l9_presecond_top_c4_climb import goals
from probe_l9_presecond_upper_c5_climb import entered_c5
from probe_l9_right_trap_stage import full_catches
from probe_l9_route_deletions import enter_level_9


def height6(root):
    child = entered_c5(root)
    for _ in range(6):
        child.step(6, 33, 33)
    return child


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
        "goals",
        goals(child),
        "full",
        full_catches(child),
        flush=True,
    )


def traverse(root, name, xs, move):
    child = height6(root)
    state((name, 0), child)
    for index, x in enumerate(xs, 1):
        child.step(6, x, 39)
        state((name, index, "CLICK", x), child)
        if child.terminal():
            return
        child.step(move)
        state((name, index, "MOVE", move), child)
        if child.terminal() or int(child.levels_completed) >= 9:
            return
    child.step(move)
    state((name, "EDGE", move), child)


def probe(env):
    enter_level_9(env)
    traverse(env, "RIGHT", (39, 45, 51, 57), 4)
    traverse(env, "LEFT", (27, 21, 15, 9, 3), 3)


if __name__ == "__main__":
    arena.run_program("bp35", probe)

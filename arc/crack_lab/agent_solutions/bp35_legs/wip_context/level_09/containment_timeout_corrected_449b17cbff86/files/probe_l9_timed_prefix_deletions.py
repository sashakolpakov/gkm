"""Delete inert pre-flip corridor actions and retest the exact returned-room exit."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from probe_l9_control_row import controls
from probe_l9_route_deletions import enter_level_9, route, step
from probe_l9_skip4_switch_choices import OMITTED, report


UP = (6, 27, 33)


def candidate(root, extra_skips):
    child = root.clone()
    for index, (_, action) in enumerate(route()):
        if index > 26:
            break
        if index not in OMITTED and index not in extra_skips:
            step(child, action)
    for _ in range(4):
        child.step(*UP)
    child.step(6, 21, 57)
    child.step(*controls(child)[0])
    child.step(*controls(child)[0])
    for action in ((6, 21, 39), 3, 3):
        step(child, action)
    child.step(*controls(child)[0])
    child.step(4)
    for _ in range(6):
        child.step(*UP)
    for col in (5, 6):
        child.step(6, 3 + 6 * col, 27)
        child.step(4)
    child.step(*controls(child)[-1])
    for action in (3, 3, 3, (6, 15, 39), 3):
        step(child, action)
    return child


def probe(env):
    enter_level_9(env)
    variants = (
        (),
        (21,),
        (23,),
        (24,),
        (25,),
        (26,),
        (23, 24),
        (25, 26),
        (23, 24, 25, 26),
        (21, 22, 23, 24, 25, 26),
    )
    for skipped in variants:
        child = candidate(env, set(skipped))
        report((skipped, "PRE_EXIT"), child)
        if child.terminal():
            continue
        child.step(6, 15, 33)
        report((skipped, "EXIT"), child)


if __name__ == "__main__":
    arena.run_program("bp35", probe)

"""Probe the control and wall-gap handoff at maximum safe outer depth."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import connected_components
from probe_l9_control_row import compact, controls
from probe_l9_stage_under_wall import enter_wall_outside


def objects(env):
    return [
        (blob.color, blob.bbox, blob.area)
        for blob in connected_components(
            env.frame(), colors=(7, 8, 12, 14, 15), min_area=2
        )
        if blob.bbox[0] < 63
    ]


def report(label, env):
    avatars = [
        blob.bbox
        for blob in connected_components(env.frame(), colors=(9, 11), min_area=3)
        if blob.bbox[0] < 63
    ]
    print(
        label,
        compact(env),
        "terminal",
        bool(env.terminal()),
        "avatars",
        avatars,
        "controls",
        controls(env),
        "objects",
        objects(env),
    )


def enter_wall_depth8(env):
    enter_wall_outside(env)
    for _ in range(8):
        env.step(6, 57, 35)


def probe(env):
    enter_wall_depth8(env)
    report("ENTRY", env)
    for control in controls(env):
        child = env.clone()
        child.step(*control)
        report(("CONTROL", control), child)
    for target, action in (((51, 27), 3), ((45, 27), 3)):
        child = env.clone()
        child.step(6, *target)
        child.step(action)
        report(("HANDOFF", target, action), child)
    for action in (3, 4, 7):
        child = env.clone()
        child.step(action)
        report(("KEY", action), child)


if __name__ == "__main__":
    arena.run_program("bp35", probe)

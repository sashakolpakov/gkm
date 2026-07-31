"""Inspect each safe band after the earliest far-right gravity flip."""

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


def enter_early_flip(env, height=3):
    enter_wall_outside(env)
    for _ in range(5):
        env.step(6, 57, 35)
    env.step(6, 3, 59)
    for _ in range(height):
        env.step(6, 57, 33)


def probe(env):
    enter_wall_outside(env)
    for _ in range(5):
        env.step(6, 57, 35)
    env.step(6, 3, 59)
    report("FLIP", env)
    for height in range(1, 4):
        env.step(6, 57, 33)
        report(("CLIMB", height), env)


if __name__ == "__main__":
    arena.run_program("bp35", probe)

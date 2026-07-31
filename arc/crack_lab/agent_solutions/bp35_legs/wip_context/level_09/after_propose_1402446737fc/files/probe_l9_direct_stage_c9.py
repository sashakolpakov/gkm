"""Test activating the lower lane-nine landing directly."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import connected_components
from probe_l9_control_row import compact, controls
from probe_l9_second_gap_cross import enter_second_gap


def objects(env):
    return [
        (blob.color, blob.bbox, blob.area)
        for blob in connected_components(
            env.frame(), colors=(7, 8, 12, 14, 15), min_area=2
        )
        if blob.bbox[0] < 63 and blob.bbox[1] >= 49
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
        "objects",
        objects(env),
    )


def probe(env):
    enter_second_gap(env, 8)
    for _ in range(6):
        env.step(6, 51, 35)
    report("SHELF", env)
    env.step(6, 57, 41)
    report("ACTIVATE_LOWER_C9", env)
    env.step(6, 57, 27)
    env.step(4)
    report("MOVE_C9", env)
    env.step(6, 57, 35)
    report("OPEN_C9", env)
    if env.terminal():
        return
    for _ in range(5):
        env.step(6, 57, 35)
    report("DEPTH5", env)
    env.step(*controls(env)[0])
    report("FLIP", env)
    for step in range(1, 9):
        env.step(3)
        report(("LEFT", step), env)
        if env.terminal():
            return


if __name__ == "__main__":
    arena.run_program("bp35", probe)

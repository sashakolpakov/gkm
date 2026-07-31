"""Probe the new left-edge control chamber reached after the compressed fall."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import connected_components
from probe_l9_control_row import compact, controls
from probe_l9_freefall_flip import enter_freefall_flip


def goals(env):
    return [
        (blob.bbox, blob.area)
        for blob in connected_components(env.frame(), colors=(7,), min_area=3)
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
        "goals",
        goals(env),
    )


def enter_second_control(env):
    enter_freefall_flip(env)
    for _ in range(9):
        env.step(3)


def enter_second_bridge(env):
    enter_second_control(env)
    for x in (57, 51, 45, 39, 33, 27, 21, 15, 9):
        env.step(6, x, 45)


def probe(env):
    enter_second_control(env)
    report("CHAMBER", env)
    direct = env.clone()
    direct.step(*controls(direct)[0])
    report("DIRECT_FLIP", direct)
    bridged = env.clone()
    for x in (57, 51, 45, 39, 33, 27, 21, 15, 9):
        bridged.step(6, x, 45)
    report("BRIDGED", bridged)
    bridged.step(*controls(bridged)[0])
    report("BRIDGED_FLIP", bridged)


if __name__ == "__main__":
    arena.run_program("bp35", probe)

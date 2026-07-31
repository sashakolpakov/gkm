"""Remove the second chamber's outer landing before flipping gravity."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import connected_components
from probe_l9_control_row import compact, controls
from probe_l9_lane9_second_walk import enter_lane9_second_control


def report(label, env):
    goals = [
        (blob.bbox, blob.area)
        for blob in connected_components(env.frame(), colors=(7,), min_area=3)
        if blob.bbox[0] < 63
    ]
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
        goals,
    )


def enter_second_freefall(env):
    enter_lane9_second_control(env)
    env.step(6, 3, 51)
    env.step(*controls(env)[0])


def probe(env):
    enter_lane9_second_control(env)
    report("CHAMBER", env)
    for target in ((3, 51), (3, 53), (3, 45), (3, 57)):
        child = env.clone()
        child.step(6, *target)
        report(("OPEN", target), child)
        child.step(*controls(child)[0])
        report(("FLIP", target), child)


if __name__ == "__main__":
    arena.run_program("bp35", probe)

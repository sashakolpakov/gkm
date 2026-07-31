"""Probe the control chamber reached after thirteen outer descents."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import connected_components
from probe_l9_control_row import compact, controls, enter_control_row


def enter_outer_controls(env):
    enter_control_row(env)
    env.step(6, 9, 3)
    for _ in range(13):
        env.step(6, 3, 33)


def summary(env):
    goals = [
        (blob.bbox, blob.area)
        for blob in connected_components(env.frame(), colors=(7,), min_area=3)
        if blob.bbox[0] < 63
    ]
    avatars = [
        (blob.bbox, blob.area)
        for blob in connected_components(env.frame(), colors=(9, 11), min_area=3)
        if blob.bbox[0] < 63
    ]
    return compact(env), "controls", controls(env), "avatars", avatars, "goals", goals


def probe(env):
    enter_outer_controls(env)
    print("ENTRY", *summary(env))
    for control in controls(env):
        child = env.clone()
        child.step(*control)
        print("CONTROL", control, *summary(child))
    for action in (3, 4, 7):
        child = env.clone()
        child.step(action)
        print("KEY", action, *summary(child))


if __name__ == "__main__":
    arena.run_program("bp35", probe)

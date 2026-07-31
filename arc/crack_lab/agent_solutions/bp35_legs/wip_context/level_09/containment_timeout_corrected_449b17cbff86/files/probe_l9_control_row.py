"""Test the nine gravity controls reached via the protected x=3 shaft."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import connected_components
from probe_l9_extra_column import compact
from probe_l9_k_room import enter_k_room


def enter_control_row(env):
    enter_k_room(env)
    env.step(6, 9, 33)
    env.step(6, 45, 33)
    for x in (39, 33, 27, 21, 15, 9, 3):
        env.step(6, x, 39)
        env.step(3)
    env.step(6, 3, 33)


def controls(env):
    return [
        (6, round(blob.centroid[1]), round(blob.centroid[0]))
        for blob in connected_components(env.frame(), colors=(8,), min_area=3)
        if blob.bbox[0] < 63
    ]


def probe(env):
    enter_control_row(env)
    print("CONTROL_ROW", compact(env), "controls", controls(env))
    for control in controls(env):
        child = env.clone()
        child.step(*control)
        print("FLIP", control, compact(child), "controls", controls(child))
        for action in (3, 4):
            moved = child.clone()
            moved.step(action)
            print("MOVE", control, action, compact(moved))


if __name__ == "__main__":
    arena.run_program("bp35", probe)

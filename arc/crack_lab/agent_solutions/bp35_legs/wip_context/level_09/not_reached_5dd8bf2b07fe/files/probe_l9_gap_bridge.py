"""Propagate the remote catch through the one-row gap in the shaft wall."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import connected_components
from probe_l9_control_row import compact
from probe_l9_outer_controls import enter_outer_controls


def report(label, env):
    avatars = [
        (blob.bbox, blob.area)
        for blob in connected_components(env.frame(), colors=(9, 11), min_area=3)
        if blob.bbox[0] < 63
    ]
    goals = [
        (blob.bbox, blob.area)
        for blob in connected_components(env.frame(), colors=(7,), min_area=3)
        if blob.bbox[0] < 63
    ]
    print(label, compact(env), "avatars", avatars, "goals", goals)


def enter_gap_bridge(env):
    enter_outer_controls(env)
    for x in (51, 45, 39, 33, 27, 21, 15, 9):
        env.step(6, x, 45)


def probe(env):
    enter_outer_controls(env)
    report("CHAMBER", env)
    for x in (51, 45, 39, 33, 27, 21, 15, 9):
        env.step(6, x, 45)
        report(("PROPAGATE", x), env)
        if env.terminal():
            return
    for action in (3, 4):
        child = env.clone()
        child.step(action)
        report(("MOVE", action), child)
    for y in (39, 45, 51):
        child = env.clone()
        child.step(6, 3, y)
        report(("EDGE_CLICK", y), child)


if __name__ == "__main__":
    arena.run_program("bp35", probe)

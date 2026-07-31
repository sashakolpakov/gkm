"""Flip gravity after staging a catch through the shaft-wall gap."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import connected_components
from probe_l9_control_row import compact, controls
from probe_l9_gap_bridge import enter_gap_bridge


def objects(env, color, area=21):
    return [
        (6, round(blob.centroid[1]), round(blob.centroid[0]))
        for blob in connected_components(env.frame(), colors=(color,), min_area=3)
        if blob.bbox[0] < 63 and blob.area == area
    ]


def report(label, env):
    avatars = [
        (blob.bbox, blob.area)
        for blob in connected_components(env.frame(), colors=(9, 11), min_area=3)
        if blob.bbox[0] < 63
    ]
    print(
        label,
        compact(env),
        "avatars",
        avatars,
        "controls",
        controls(env),
        "catches",
        objects(env, 15),
        "blocks",
        objects(env, 14),
        "goals",
        objects(env, 7, 5),
    )


def enter_gap_landing(env):
    enter_gap_bridge(env)
    env.step(6, 15, 3)


def probe(env):
    enter_gap_bridge(env)
    report("BRIDGED", env)
    for control in controls(env):
        child = env.clone()
        child.step(*control)
        report(("FLIP", control), child)
        for action in (3, 4):
            moved = child.clone()
            moved.step(action)
            report(("MOVE", action), moved)
        for catch in objects(child, 15):
            if catch[2] in (27, 33, 39, 45):
                clicked = child.clone()
                clicked.step(*catch)
                report(("CATCH", catch), clicked)


if __name__ == "__main__":
    arena.run_program("bp35", probe)

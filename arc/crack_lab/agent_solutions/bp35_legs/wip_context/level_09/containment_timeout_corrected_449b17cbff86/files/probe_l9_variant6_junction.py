"""Probe the junction reached by aligning the generated lane-six column."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import connected_components
from probe_l9_control_row import compact, controls
from probe_l9_gap_flip import enter_gap_landing
from probe_l9_route_variants import build_variant


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


def enter_variant6_junction(env):
    enter_gap_landing(env)
    env.step(6, 3, 35)
    routed = build_variant(env, 6)
    for _ in range(5):
        routed.step(6, 57, 35)
    routed.step(*controls(routed)[0])
    for _ in range(3):
        routed.step(3)
    return routed


def probe(env):
    junction = enter_variant6_junction(env)
    report("JUNCTION", junction)
    candidates = (
        (3,),
        (4,),
        (7,),
        (6, 39, 33),
        (6, 33, 39),
        (6, 45, 39),
        (6, 39, 45),
    )
    for action in candidates:
        child = junction.clone()
        child.step(*action)
        report(("ACTION", action), child)


if __name__ == "__main__":
    arena.run_program("bp35", probe)

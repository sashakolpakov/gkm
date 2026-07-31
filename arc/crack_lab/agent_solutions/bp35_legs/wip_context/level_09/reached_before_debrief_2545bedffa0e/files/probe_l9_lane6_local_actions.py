"""Test context-dependent clicks beside and below the post-shelf avatar."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import connected_components
from probe_l9_control_row import compact
from probe_l9_lane6_continue import enter_lane6_shelf


def objects(env):
    return [
        (blob.color, blob.bbox, blob.area)
        for blob in connected_components(
            env.frame(), colors=(7, 8, 12, 14, 15), min_area=2
        )
        if blob.bbox[0] < 63 and 20 <= blob.bbox[0] <= 45
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
    enter_lane6_shelf(env)
    report("ENTRY", env)
    for target in ((45, 27), (45, 33), (45, 35), (51, 35), (57, 35)):
        child = env.clone()
        child.step(6, *target)
        report(("CLICK", target), child)
        for action in (3, 4):
            moved = child.clone()
            moved.step(action)
            report(("THEN", target, action), moved)
        clicked_again = child.clone()
        clicked_again.step(6, *target)
        report(("AGAIN", target), clicked_again)
    for action in (3, 4):
        child = env.clone()
        child.step(action)
        report(("KEY", action), child)


if __name__ == "__main__":
    arena.run_program("bp35", probe)

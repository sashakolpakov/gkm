"""Turn down from column two before the second chamber collision."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import connected_components
from probe_l9_control_row import compact, controls
from probe_l9_second_direct_walk import enter_second_walk


def report(label, env):
    avatars = [
        blob.bbox
        for blob in connected_components(env.frame(), colors=(9, 11), min_area=3)
        if blob.bbox[0] < 63
    ]
    goals = [
        (blob.bbox, blob.area)
        for blob in connected_components(env.frame(), colors=(7,), min_area=3)
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


def enter_second_down(env):
    enter_second_walk(env, 2)
    env.step(6, 15, 35)


def probe(env):
    enter_second_walk(env, 2)
    report("TURN", env)
    for action in ((6, 15, 35), (6, 21, 35), (6, 15, 27), (3,), (4,), (7,)):
        child = env.clone()
        child.step(*action)
        report(("ACTION", action), child)


if __name__ == "__main__":
    arena.run_program("bp35", probe)

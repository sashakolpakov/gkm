"""Turn vertically from column six after the optimized second flip."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import connected_components
from probe_l9_control_row import compact, controls
from probe_l9_fast_prefix import enter_lane9_second_control_fast


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


def enter_fast_second_turn(env):
    enter_lane9_second_control_fast(env)
    env.step(*controls(env)[0])
    for _ in range(6):
        env.step(4)


def probe(env):
    enter_fast_second_turn(env)
    report("TURN", env)
    for action in (
        (6, 39, 35),
        (6, 39, 29),
        (6, 45, 35),
        (6, 39, 41),
        (3,),
        (4,),
        (7,),
    ):
        child = env.clone()
        child.step(*action)
        report(("ACTION", action), child)


if __name__ == "__main__":
    arena.run_program("bp35", probe)

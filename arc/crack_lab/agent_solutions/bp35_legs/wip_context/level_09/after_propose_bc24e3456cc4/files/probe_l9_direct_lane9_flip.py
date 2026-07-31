"""Flip at the deep control after the direct lane-nine descent."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import connected_components
from probe_l9_control_row import compact, controls
from probe_l9_second_gap_cross import enter_second_gap


def avatar(env):
    return [
        blob.bbox
        for blob in connected_components(env.frame(), colors=(9, 11), min_area=3)
        if blob.bbox[0] < 63
    ]


def enter_lane9_flip(env):
    enter_second_gap(env, 9)
    for _ in range(7):
        env.step(6, 57, 35)
    env.step(*controls(env)[0])


def probe(env):
    enter_lane9_flip(env)
    print("FLIP", compact(env), "terminal", bool(env.terminal()))
    for step in range(1, 13):
        env.step(3)
        print(
            "LEFT",
            step,
            compact(env),
            "terminal",
            bool(env.terminal()),
            "avatar",
            avatar(env),
        )
        if env.terminal() or int(env.levels_completed) >= 9:
            return


if __name__ == "__main__":
    arena.run_program("bp35", probe)

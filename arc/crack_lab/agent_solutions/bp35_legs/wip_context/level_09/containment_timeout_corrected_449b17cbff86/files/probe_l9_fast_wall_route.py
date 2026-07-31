"""Use lane eight to skip the off shelf toggle before the wall route."""

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


def enter_wall_outside_fast(env):
    enter_second_gap(env, 8)
    for _ in range(6):
        env.step(6, 51, 35)
    for col in (6, 7, 8):
        env.step(6, 3 + 6 * col, 41)
    env.step(6, 57, 27)
    env.step(4)
    env.step(6, 57, 35)


def probe(env):
    enter_wall_outside_fast(env)
    print(
        "OUTSIDE",
        compact(env),
        "terminal",
        bool(env.terminal()),
        "avatar",
        avatar(env),
    )
    for _ in range(5):
        env.step(6, 57, 35)
    print("DEPTH5", compact(env), "controls", controls(env))
    env.step(*controls(env)[0])
    print("FLIP", compact(env), "terminal", bool(env.terminal()))
    for step in range(1, 9):
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
        if env.terminal():
            return


if __name__ == "__main__":
    arena.run_program("bp35", probe)

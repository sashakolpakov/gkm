"""Replay the verified route with the first outer flip four actions earlier."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import connected_components
from probe_l9_control_row import compact, controls, enter_control_row


def avatar(env):
    return [
        blob.bbox
        for blob in connected_components(env.frame(), colors=(9, 11), min_area=3)
        if blob.bbox[0] < 63
    ]


def enter_outer_controls_fast(env):
    enter_control_row(env)
    env.step(6, 9, 3)
    for _ in range(8):
        env.step(6, 3, 33)
    env.step(6, 3, 59)


def enter_gap_landing_fast(env):
    enter_outer_controls_fast(env)
    for x in (51, 45, 39, 33, 27, 21, 15, 9):
        env.step(6, x, 45)
    env.step(6, 15, 3)


def enter_lane9_second_control_fast(env):
    enter_gap_landing_fast(env)
    env.step(6, 3, 35)
    for col in range(1, 10):
        env.step(6, 3 + 6 * col, 27)
        env.step(4)
    for _ in range(7):
        env.step(6, 57, 35)
    env.step(*controls(env)[0])
    for _ in range(9):
        env.step(3)


def probe(env):
    enter_lane9_second_control_fast(env)
    print("SECOND_CHAMBER", compact(env), "controls", controls(env))
    env.step(*controls(env)[0])
    print("SECOND_FLIP", compact(env), "terminal", bool(env.terminal()))
    for step in range(1, 13):
        env.step(4)
        print(
            "RIGHT",
            step,
            compact(env),
            "terminal",
            bool(env.terminal()),
            "avatar",
            avatar(env),
            "controls",
            controls(env),
        )
        if env.terminal() or int(env.levels_completed) >= 9:
            return


if __name__ == "__main__":
    arena.run_program("bp35", probe)

"""Cross the second chamber with the direct lane-nine catch layout."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import connected_components
from probe_l9_control_row import compact, controls
from probe_l9_direct_lane9_flip import enter_lane9_flip


def avatar(env):
    return [
        blob.bbox
        for blob in connected_components(env.frame(), colors=(9, 11), min_area=3)
        if blob.bbox[0] < 63
    ]


def enter_lane9_second_control(env):
    enter_lane9_flip(env)
    for _ in range(9):
        env.step(3)


def probe(env):
    enter_lane9_second_control(env)
    print("CHAMBER", compact(env), "controls", controls(env))
    env.step(*controls(env)[0])
    print("FLIP", compact(env), "terminal", bool(env.terminal()))
    for step in range(1, 11):
        env.step(4)
        print(
            "RIGHT",
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

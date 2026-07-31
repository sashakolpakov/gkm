"""Climb the far-right catch after the depth-eight gravity reversal."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import connected_components
from probe_l9_control_row import compact, controls
from probe_l9_second_descent import component_at
from probe_l9_wall_depth8 import enter_wall_depth8


def goals(env):
    return [
        (blob.bbox, blob.area)
        for blob in connected_components(env.frame(), colors=(7,), min_area=3)
        if blob.bbox[0] < 63
    ]


def enter_after_wall_flip(env):
    enter_wall_depth8(env)
    env.step(6, 3, 41)


def probe(env):
    enter_after_wall_flip(env)
    print(
        "FLIPPED",
        compact(env),
        "above",
        component_at(env, 57, 33),
        "controls",
        controls(env),
        "goals",
        goals(env),
    )
    for height in range(1, 26):
        above = component_at(env, 57, 33)
        if not above or above[0] != 15 or above[1] != 21:
            print("STOP", height - 1, "above", above)
            return
        env.step(6, 57, 33)
        print(
            "CLIMB",
            height,
            compact(env),
            "above",
            component_at(env, 57, 33),
            "controls",
            controls(env),
            "goals",
            goals(env),
        )
        if env.terminal() or int(env.levels_completed) >= 9:
            return


if __name__ == "__main__":
    arena.run_program("bp35", probe)

"""Descend the generated lane-nine catch immediately after the first wall."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import connected_components
from probe_l9_control_row import compact, controls
from probe_l9_second_gap_cross import enter_second_gap


def goals(env):
    return [
        (blob.bbox, blob.area)
        for blob in connected_components(env.frame(), colors=(7,), min_area=3)
        if blob.bbox[0] < 63
    ]


def probe(env):
    enter_second_gap(env, 9)
    print(
        "LANE9",
        compact(env),
        "terminal",
        bool(env.terminal()),
        "controls",
        controls(env),
        "goals",
        goals(env),
    )
    for depth in range(1, 21):
        env.step(6, 57, 35)
        print(
            "DESCEND",
            depth,
            compact(env),
            "terminal",
            bool(env.terminal()),
            "controls",
            controls(env),
            "goals",
            goals(env),
        )
        if env.terminal() or int(env.levels_completed) >= 9:
            return


if __name__ == "__main__":
    arena.run_program("bp35", probe)

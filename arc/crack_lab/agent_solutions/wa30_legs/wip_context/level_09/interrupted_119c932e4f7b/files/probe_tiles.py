"""Exact small-tile views for contact-state verification."""

import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import ACTION_NAME, color_counts


SYMBOL = {0: "0", 1: "."}


def crop(frame, r0, c0, r1, c1):
    a = np.asarray(frame)
    return "/".join(
        "".join(SYMBOL.get(int(value), f"{int(value):X}") for value in a[row, c0:c1])
        for row in range(r0, r1)
    )


def run_path(env, name, path):
    clone = env.clone()
    print(name, "START", crop(clone.frame(), 32, 24, 44, 44))
    for action in path:
        clone.step(action)
        print(
            ACTION_NAME[action],
            "counts",
            color_counts(clone.frame()),
            "zone",
            crop(clone.frame(), 32, 24, 44, 44),
            "level",
            clone.levels_completed,
        )


def probe(env):
    run_path(env, "below_then_use_leave", [1, 1, 5, 5, 2])
    run_path(env, "left_then_face_use_leave", [3, 1, 1, 1, 4, 5, 3])
    run_path(env, "right_then_face_use_leave", [4, 1, 1, 1, 3, 5, 4])


arena.run_program("wa30", probe)

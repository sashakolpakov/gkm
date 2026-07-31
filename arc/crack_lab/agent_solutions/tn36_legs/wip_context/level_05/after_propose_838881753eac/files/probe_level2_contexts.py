"""Disambiguate the two level-2 context codes."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A
import numpy as np

from legs import make_small_segments_color_5_and_submit


ROWS = (33, 36, 39, 42, 45, 48)
LEFT_COLUMNS = (8, 13, 18, 23)
RIGHT_COLUMNS = (39, 44, 49, 54)


def panel(frame, columns):
    return tuple("".join(str(int(frame[row, col])) for col in columns) for row in ROWS)


def state(env):
    frame = np.asarray(env.frame())
    yellow = np.argwhere((frame[:32, :31] == 4))
    return {
        "left": panel(frame, LEFT_COLUMNS),
        "right": panel(frame, RIGHT_COLUMNS),
        "selected": (int(frame[54, 11]), int(frame[54, 21])),
        "yellow_bbox": tuple(int(value) for value in (*yellow.min(axis=0), *yellow.max(axis=0))),
    }


def observe(env):
    make_small_segments_color_5_and_submit(env)
    print("base", state(env))
    for name, x in (("left_selected", 11), ("up", 21)):
        clone = env.clone()
        clone.step(6, x, 58)
        print(name, state(clone))


levels, path, error = A.run_program("tn36", observe)
print("probe_result", {"levels": levels, "moves": len(path), "error": error})

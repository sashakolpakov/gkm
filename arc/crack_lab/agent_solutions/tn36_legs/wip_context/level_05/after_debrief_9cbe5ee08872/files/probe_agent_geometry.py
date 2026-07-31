"""Exact compact masks for the cyan board objects on levels 2 and 3."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A
import numpy as np

from legs import (
    make_small_segments_color_5_and_submit,
    turn_on_outer_rows_of_right_segment_panel_and_submit,
)
from perception import connected_components


def masks(env, label):
    frame = np.asarray(env.frame())
    print(label)
    for blob in connected_components(frame, colors=(11,), min_area=4):
        if blob.bbox[0] >= 32:
            continue
        r0, c0, r1, c1 = blob.bbox
        mask = tuple(
            "".join("#" if int(frame[row, col]) == 11 else "." for col in range(c0, c1 + 1))
            for row in range(r0, r1 + 1)
        )
        print({"bbox": blob.bbox, "area": blob.area, "mask": mask})


def observe(env):
    make_small_segments_color_5_and_submit(env)
    masks(env, "level_2")
    turn_on_outer_rows_of_right_segment_panel_and_submit(env)
    masks(env, "level_3")


levels, path, error = A.run_program("tn36", observe)
print("probe_result", {"levels": levels, "moves": len(path), "error": error})

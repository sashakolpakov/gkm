"""Reproduce the solved tutorial frames without touching campaign state."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A

from legs import (
    make_small_segments_color_5_and_submit,
    turn_on_outer_rows_of_right_segment_panel_and_submit,
)
from perception import color_counts, connected_components


def summary(env, label):
    grouped = {}
    for blob in connected_components(env.frame(), min_area=3):
        grouped.setdefault(blob.color, []).append((blob.bbox, blob.area))
    print(label, {"completed": env.levels_completed, "colors": color_counts(env.frame())})
    for color, blobs in grouped.items():
        print("components", color, blobs)


def observe(env):
    summary(env, "level_1_entry")
    make_small_segments_color_5_and_submit(env)
    summary(env, "level_2_entry")
    turn_on_outer_rows_of_right_segment_panel_and_submit(env)
    summary(env, "level_3_entry")


levels, path, error = A.run_program("tn36", observe)
print("probe_result", {"levels": levels, "moves": len(path), "error": error})

"""Reproduce level-3 selector examples and their known route vocabulary."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A

from legs import (
    click_largest_color_9_submit_disc,
    make_small_segments_color_5,
    turn_on_outer_rows_of_right_segment_panel,
)
from perception import arr, connected_components


ROWS = (33, 36, 39, 42, 45, 48)


def shape(frame, blob):
    r0, c0, r1, c1 = blob.bbox
    return tuple(
        "".join("#" if int(frame[row, col]) == blob.color else "." for col in range(c0, c1 + 1))
        for row in range(r0, r1 + 1)
    )


def panel_bits(frame):
    columns = sorted(
        {
            int(round(blob.centroid[1]))
            for blob in connected_components(frame, colors=(1, 5), min_area=3)
            if blob.area == 3 and 1 in blob.size and blob.centroid[1] < 31
        }
    )
    return tuple(
        tuple(int(frame[row, col]) == 5 for row in ROWS) for col in columns
    )


def demo(frame):
    return [
        (blob.bbox, blob.area, shape(frame, blob))
        for blob in connected_components(frame, colors=(4,))
        if blob.bbox[2] < 32 and blob.bbox[3] < 32
    ]


def observe(env):
    make_small_segments_color_5(env)
    click_largest_color_9_submit_disc(env)
    frame = turn_on_outer_rows_of_right_segment_panel(env)
    click_largest_color_9_submit_disc(env, frame)
    print("entry_level", env.levels_completed + 1)
    for name, col in (("a", 5), ("b", 15), ("c", 25), ("d", 35)):
        clone = env.clone()
        clone.step(6, col, 58)
        current = arr(clone.frame())
        print(name, {"demo": demo(current), "bits": panel_bits(current)})


levels, path, error = A.run_program("tn36", observe)
print("probe_result", {"levels": levels, "moves": len(path), "error": error})

"""Compare selectable protocol examples at reproducible level entries."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A

from legs import (
    encode_reacquisition_route_through_barrier_and_submit,
    make_small_segments_color_5_and_submit,
    turn_on_outer_rows_of_right_segment_panel_and_submit,
)
from perception import arr, connected_components


def normalized_shape(frame, blob):
    r0, c0, r1, c1 = blob.bbox
    return tuple(
        "".join("#" if int(frame[row, col]) == blob.color else "." for col in range(c0, c1 + 1))
        for row in range(r0, r1 + 1)
    )


def segment_geometry(frame):
    blobs = [
        blob
        for blob in connected_components(frame, colors=(1, 5), min_area=3)
        if blob.area == 3 and 1 in blob.size
    ]
    rows = sorted({int(round(blob.centroid[0])) for blob in blobs})
    left_cols = sorted(
        {
            int(round(blob.centroid[1]))
            for blob in blobs
            if blob.centroid[1] < 31
        }
    )
    return rows, left_cols


def state(node):
    frame = arr(node.frame())
    rows, columns = segment_geometry(frame)
    bits = tuple(
        "".join("1" if int(frame[row, col]) == 5 else "0" for row in rows)
        for col in columns
    )
    demo = tuple(
        (blob.color, blob.bbox, blob.area, normalized_shape(frame, blob))
        for blob in connected_components(frame, colors=(4,))
        if blob.bbox[1] < 31
    )
    selector = tuple(
        sum(int(value) == 9 for value in frame[54:63, col - 4 : col + 5].flat)
        for col in (5, 15, 25, 35)
    )
    return {"demo4": demo, "left_bits": bits, "selector9": selector}


def button_patterns(frame):
    symbols = {0: "0", 2: "2", 5: ".", 9: "9", 11: "B"}
    result = {}
    for center in (5, 15, 25, 35):
        region = frame[55:62, center - 3 : center + 4]
        result[center] = tuple(
            "".join(symbols.get(int(value), str(int(value))) for value in row)
            for row in region
        )
    return result


def observe(env):
    make_small_segments_color_5_and_submit(env)
    turn_on_outer_rows_of_right_segment_panel_and_submit(env)
    print("entry_level", env.levels_completed + 1)
    print("buttons", button_patterns(arr(env.frame())))
    print("entry", state(env))
    for center in (5, 15, 25, 35):
        clone = env.clone()
        clone.step(6, center, 58)
        print("select", center, state(clone))

    encode_reacquisition_route_through_barrier_and_submit(env)
    print("entry_level", env.levels_completed + 1)
    print("buttons", button_patterns(arr(env.frame())))
    print("entry", state(env))
    for center in (5, 15, 25, 35):
        clone = env.clone()
        clone.step(6, center, 58)
        print("select", center, state(clone))


levels, path, error = A.run_program("tn36", observe)
print("probe_result", {"levels": levels, "moves": len(path), "error": error})

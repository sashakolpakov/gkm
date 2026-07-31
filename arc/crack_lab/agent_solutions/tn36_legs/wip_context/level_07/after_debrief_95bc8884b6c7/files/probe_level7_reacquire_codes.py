"""Learn the level-7 checkpoint filler glyph from a staged root."""

import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A

import solve
from legs import (
    click_largest_color_9_submit_disc,
    learn_direction_protocol_from_selector,
)
from perception import arr, connected_components


def panel_geometry(frame):
    segments = [
        blob
        for blob in connected_components(frame, colors=(1, 5), min_area=3)
        if blob.area == 3
        and 1 in blob.size
        and blob.centroid[0] > len(frame) / 2
        and blob.centroid[1] > len(frame[0]) / 2
    ]
    rows = sorted({int(round(blob.centroid[0])) for blob in segments})
    cols = sorted({int(round(blob.centroid[1])) for blob in segments})
    return rows, cols


def set_program(env, rows, cols, program, codes):
    frame = arr(env.frame()).copy()
    for col, symbol in zip(cols, program):
        for row_index, row in enumerate(rows):
            is_on = int(frame[row][col]) == 5
            should_be_on = row_index in codes[symbol]
            if is_on != should_be_on:
                env.step(6, col, row)


def avatar_objects(frame):
    return tuple(
        (blob.bbox, blob.area)
        for blob in connected_components(frame, colors=(11,), min_area=9)
        if blob.bbox[2] < 32 and blob.bbox[1] >= 32
    )


def observe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)
    solve.solve(env)

    codes = learn_direction_protocol_from_selector(env)
    rows, cols = panel_geometry(env.frame())
    set_program(env, rows, cols, "UURRRD", codes)
    click_largest_color_9_submit_disc(env)
    staged = avatar_objects(env.frame())
    if (24, 53, 27, 56) not in {bbox for bbox, _ in staged}:
        raise RuntimeError(("first checkpoint failed", staged))
    root_frame = arr(env.frame()).copy()

    def outcome(mask):
        clone = env.clone()
        active_codes = dict(codes)
        active_codes["P"] = tuple(
            row_index for row_index in range(6) if mask & (1 << row_index)
        )
        set_program(clone, rows, cols, "PPUUUU", active_codes)
        click_largest_color_9_submit_disc(clone)
        frame = arr(clone.frame()).copy()
        return frame, avatar_objects(frame), clone.levels_completed

    baseline_frame, baseline_objects, baseline_level = outcome(9)
    hits = []
    for mask in range(64):
        frame, objects, level = outcome(mask)
        at_upper_checker = (8, 53, 11, 56) in {bbox for bbox, _ in objects}
        if at_upper_checker or level > env.levels_completed:
            hits.append(
                {
                    "mask": mask,
                    "rows": tuple(i for i in range(6) if mask & (1 << i)),
                    "objects": objects,
                    "level": level,
                }
            )
    repeated_frame, repeated_objects, repeated_level = outcome(9)
    print(
        "reacquire_search",
        {
            "staged": staged,
            "root_unchanged": bool((root_frame == arr(env.frame())).all()),
            "baseline": (baseline_objects, baseline_level),
            "repeat_same": bool((baseline_frame == repeated_frame).all()),
            "repeated": (repeated_objects, repeated_level),
            "hits": hits,
        },
    )


levels, path, error = A.run_program("tn36", observe)
print("probe_result", {"levels": levels, "moves": len(path), "error": error})

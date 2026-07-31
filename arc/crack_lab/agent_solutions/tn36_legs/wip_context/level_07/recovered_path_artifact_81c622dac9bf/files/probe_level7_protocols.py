"""Probe level-7 protocol effects on independent pristine clones."""

import json
import sys
from collections import Counter

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A

import solve
from legs import (
    PROTOCOL_ROWS,
    click_largest_color_9_submit_disc,
    learn_direction_protocol_from_selector,
)
from perception import arr, connected_components, frame_delta


def transitions(before, after):
    changed = arr(before) != arr(after)
    return tuple(
        sorted(
            Counter(
                zip(
                    (int(v) for v in arr(before)[changed]),
                    (int(v) for v in arr(after)[changed]),
                )
            ).items()
        )
    )


def objects(frame):
    return tuple(
        (blob.color, blob.bbox, blob.area)
        for blob in connected_components(
            frame,
            colors=(4, 11, 12, 13, 14, 15),
            min_area=2,
        )
        if blob.bbox[2] < 32
        and (blob.color != 4 or blob.bbox[1] < 32)
    )


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


def run_program(root, program, codes):
    clone = root.clone()
    start = arr(clone.frame()).copy()
    rows, cols = panel_geometry(start)
    for col, symbol in zip(cols, program):
        if symbol is None:
            continue
        for row_index in codes[symbol]:
            clone.step(6, col, rows[row_index])
    configured = arr(clone.frame()).copy()
    click_largest_color_9_submit_disc(clone)
    final = arr(clone.frame()).copy()
    return {
        "program": "".join(symbol or "-" for symbol in program),
        "level_delta": clone.levels_completed - root.levels_completed,
        "config_delta": frame_delta(start, configured)["count"],
        "submit_delta": frame_delta(configured, final),
        "transitions": transitions(configured, final),
        "objects_before": objects(configured),
        "objects_after": objects(final),
    }


def observe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)
    solve.solve(env)

    learner = env.clone()
    learned = learn_direction_protocol_from_selector(learner)
    print("learned_directions", learned)
    codes = dict(PROTOCOL_ROWS)
    codes.update(learned or {})

    scenarios = [([None] * 6)]
    for symbol in ("D", "U", "L", "R", "M", "X", "A", "B"):
        scenarios.append([symbol] + [None] * 5)
        scenarios.append([None] * 5 + [symbol])
    for symbol in ("D", "U", "L", "R", "M", "X", "A", "B"):
        scenarios.append([symbol] * 6)

    for program in scenarios:
        print("effect", run_program(env, program, codes))


levels, path, error = A.run_program("tn36", observe)
print("probe_result", {"levels": levels, "moves": len(path), "error": error})

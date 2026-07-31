"""Probe contextual board clicks after reaching level-7's first checker."""

import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A

import solve
from legs import (
    click_largest_color_9_submit_disc,
    learn_direction_protocol_from_selector,
)
from perception import arr, connected_components, frame_delta


def panel_geometry(frame):
    segments = [
        blob
        for blob in connected_components(frame, colors=(1, 5), min_area=3)
        if blob.area == 3
        and 1 in blob.size
        and blob.centroid[0] > len(frame) / 2
        and blob.centroid[1] > len(frame[0]) / 2
    ]
    return (
        sorted({int(round(blob.centroid[0])) for blob in segments}),
        sorted({int(round(blob.centroid[1])) for blob in segments}),
    )


def set_program(env, rows, cols, program, codes):
    frame = arr(env.frame()).copy()
    for col, symbol in zip(cols, program):
        for row_index, row in enumerate(rows):
            if (int(frame[row][col]) == 5) != (row_index in codes[symbol]):
                env.step(6, col, row)


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
    staged = arr(env.frame()).copy()

    effects = []
    for board_row in range(7):
        for board_col in range(7):
            x = 34 + 4 * board_col
            y = 5 + 4 * board_row
            clone = env.clone()
            clone.step(6, x, y)
            delta = frame_delta(staged, clone.frame())
            if delta["count"]:
                effects.append(
                    {
                        "cell": (board_row, board_col),
                        "at": (x, y),
                        "delta": delta,
                    }
                )
    print("staged_click_effects", effects)


levels, path, error = A.run_program("tn36", observe)
print("probe_result", {"levels": levels, "moves": len(path), "error": error})

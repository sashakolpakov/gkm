"""Observe live board previews for one glyph at the staged checkpoint."""

import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A

import solve
from legs import (
    PROTOCOL_ROWS,
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


def board_components(frame):
    return tuple(
        (blob.color, blob.bbox, blob.area)
        for blob in connected_components(
            frame,
            colors=range(7, 16),
            min_area=2,
        )
        if blob.bbox[2] < 32 and blob.bbox[1] >= 32
    )


def observe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)
    solve.solve(env)

    codes = dict(PROTOCOL_ROWS)
    codes.update(learn_direction_protocol_from_selector(env))
    codes["-"] = ()
    rows, cols = panel_geometry(env.frame())
    set_program(env, rows, cols, "UURRRD", codes)
    click_largest_color_9_submit_disc(env)
    root_frame = arr(env.frame()).copy()
    root_board = root_frame[4:32, 33:61]

    effects = []
    for symbol in "DULRMXAB":
        for index in range(6):
            program = "-" * index + symbol + "-" * (5 - index)
            clone = env.clone()
            set_program(clone, rows, cols, program, codes)
            frame = arr(clone.frame())
            board_delta = int((root_board != frame[4:32, 33:61]).sum())
            if board_delta:
                effects.append(
                    {
                        "program": program,
                        "board_delta": board_delta,
                        "components": board_components(frame),
                    }
                )
    print(
        "staged_previews",
        {
            "root_unchanged": bool((root_frame == arr(env.frame())).all()),
            "effects": effects,
        },
    )


levels, path, error = A.run_program("tn36", observe)
print("probe_result", {"levels": levels, "moves": len(path), "error": error})

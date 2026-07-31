"""Probe the six visible level-5 controls on pristine clones."""

import json
import sys
from collections import Counter

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A

from perception import arr, color_counts, connected_components


BUTTONS = {"tiny": 5, "square": 15, "elbow": 25, "plus": 35, "white": 45, "submit": 57}
PANEL_ROWS = (33, 36, 39, 42, 45, 48)
LEFT_COLS = (11, 16, 21)
RIGHT_COLS = (34, 39, 44, 49, 54, 59)


def region_blobs(frame, colors, row_limit, col_limit):
    return [
        (blob.color, blob.bbox, blob.area)
        for blob in connected_components(frame, colors=colors)
        if blob.bbox[2] < row_limit and blob.bbox[3] < col_limit
    ]


def summarize(env):
    frame = arr(env.frame())
    return {
        "level": env.levels_completed,
        "left_objects": region_blobs(frame, (4, 11, 15), 32, 32),
        "board_11": [
            (blob.bbox, blob.area)
            for blob in connected_components(frame, colors=(11,))
            if blob.bbox[2] < 32 and blob.bbox[1] > 31
        ],
        "board_15": [
            (blob.bbox, blob.area)
            for blob in connected_components(frame, colors=(15,))
            if blob.bbox[2] < 32 and blob.bbox[1] > 31
        ],
        "left_panel": tuple(
            tuple(int(frame[row, col]) for row in PANEL_ROWS) for col in LEFT_COLS
        ),
        "right_panel": tuple(
            tuple(int(frame[row, col]) for row in PANEL_ROWS) for col in RIGHT_COLS
        ),
        "button_border": tuple(int(frame[54, col]) for col in BUTTONS.values()),
        "colors": color_counts(frame),
    }


def observe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)

    print("base", summarize(env))
    sequences = [
        (name, [(6, col, 58)]) for name, col in BUTTONS.items()
    ] + [
        ("square_twice", [(6, 15, 58), (6, 15, 58)]),
        ("square_elbow", [(6, 15, 58), (6, 25, 58)]),
        ("elbow_square", [(6, 25, 58), (6, 15, 58)]),
        ("plus_white", [(6, 35, 58), (6, 45, 58)]),
        ("white_plus", [(6, 45, 58), (6, 35, 58)]),
    ]
    for name, actions in sequences:
        clone = env.clone()
        for action in actions:
            clone.step(*action)
        print(name, summarize(clone))


levels, path, error = A.run_program("tn36", observe)
print("probe_result", {"levels": levels, "moves": len(path), "error": error})

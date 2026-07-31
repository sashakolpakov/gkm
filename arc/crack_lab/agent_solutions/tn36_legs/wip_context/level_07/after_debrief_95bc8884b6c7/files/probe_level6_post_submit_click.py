"""Test coordinate interactions after routing the level-6 agent to a checker."""

import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A

from legs import click_largest_color_9_submit_disc, find_right_segment_panel
from perception import arr, connected_components, frame_delta


CODES = {
    "D": (0, 1),
    "U": (0, 5),
    "R": (1,),
    "L": (0,),
}
PROGRAM = sys.argv[1]
CLICK_X = int(sys.argv[2])
CLICK_Y = int(sys.argv[3])
with open("checkpoint.json") as stream:
    CHECKPOINT = json.load(stream)


def board_components(frame, color):
    return tuple(
        (blob.bbox, blob.area)
        for blob in connected_components(frame, colors=(color,), min_area=4)
        if blob.bbox[2] < 32 and blob.bbox[1] > 32
    )


def observe(env):
    for action in CHECKPOINT["final_path"]:
        env.step(action)
    entry = arr(env.frame()).copy()
    _, rows, cols = find_right_segment_panel(entry)
    for col, symbol in zip(cols, PROGRAM):
        for row_index in CODES[symbol]:
            env.step(6, col, rows[row_index])
    click_largest_color_9_submit_disc(env)
    routed = arr(env.frame()).copy()
    level = env.levels_completed
    env.step(6, CLICK_X, CLICK_Y)
    clicked = arr(env.frame()).copy()
    print(
        "post_click",
        {
            "program": PROGRAM,
            "click": (CLICK_X, CLICK_Y),
            "delta": frame_delta(routed, clicked),
            "level_delta": env.levels_completed - level,
            "cyan": board_components(clicked, 11),
            "white": board_components(clicked, 15),
        },
    )


levels, path, error = A.run_program("tn36", observe)
print("probe_result", {"levels": levels, "moves": len(path), "error": error})

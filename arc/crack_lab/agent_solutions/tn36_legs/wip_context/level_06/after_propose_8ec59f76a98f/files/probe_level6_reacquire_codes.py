"""Search one unknown reacquisition glyph before the forced route LLLLD."""

import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A

from legs import click_largest_color_9_submit_disc, find_right_segment_panel
from perception import arr, connected_components


with open("checkpoint.json") as stream:
    CHECKPOINT = json.load(stream)


def board_components(frame, color):
    return tuple(
        (blob.bbox, blob.area)
        for blob in connected_components(frame, colors=(color,), min_area=4)
        if blob.bbox[2] < 32 and blob.bbox[1] > 32
    )


def run(mask_value):
    observation = {}

    def observe(env):
        for action in CHECKPOINT["final_path"]:
            env.step(action)
        entry = arr(env.frame()).copy()
        _, rows, cols = find_right_segment_panel(entry)
        masks = (
            tuple(index for index in range(6) if mask_value & (1 << index)),
            (0,),
            (0,),
            (0,),
            (0,),
            (0, 1),
        )
        for col, row_indices in zip(cols, masks):
            for row_index in row_indices:
                env.step(6, col, rows[row_index])
        click_largest_color_9_submit_disc(env)
        frame = arr(env.frame())
        observation.update(
            {
                "mask": masks[0],
                "level": env.levels_completed,
                "board_delta": int((entry[:32, 32:] != frame[:32, 32:]).sum()),
                "cyan": board_components(frame, 11),
                "white": board_components(frame, 15),
            }
        )

    levels, path, error = A.run_program("tn36", observe)
    observation.update({"result_levels": levels, "moves": len(path), "error": error})
    return observation


for value in range(64):
    result = run(value)
    if result["level"] >= 6 or result["board_delta"] > 0:
        print("candidate", result, flush=True)
    if value % 8 == 7:
        print("progress", value + 1, flush=True)
    if result["level"] >= 6:
        break

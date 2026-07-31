"""Run each supplied level-6 program in a fresh Arena, never a sibling clone."""

import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A

from legs import click_largest_color_9_submit_disc, find_right_segment_panel
from perception import arr, connected_components


CODES = {
    "N": (),
    "D": (0, 1),
    "U": (0, 5),
    "R": (1,),
    "L": (0,),
    "X": (3,),
    "M": (0, 3),
    "A": (0, 2),
    "B": (0, 1, 2, 3, 4, 5),
}
with open("checkpoint.json") as stream:
    CHECKPOINT = json.load(stream)


def board_components(frame, color):
    return tuple(
        (
            blob.bbox,
            blob.area,
            tuple(
                "".join(
                    "#" if int(frame[row][col]) == color else "."
                    for col in range(blob.bbox[1], blob.bbox[3] + 1)
                )
                for row in range(blob.bbox[0], blob.bbox[2] + 1)
            ),
        )
        for blob in connected_components(frame, colors=(color,), min_area=4)
        if blob.bbox[2] < 32 and blob.bbox[1] > 32
    )


def run(program):
    observation = {}

    def observe(env):
        for action in CHECKPOINT["final_path"]:
            env.step(action)
        entry = arr(env.frame()).copy()
        _, rows, cols = find_right_segment_panel(entry)
        for col, symbol in zip(cols, program):
            for row_index in CODES[symbol]:
                env.step(6, col, rows[row_index])
        click_largest_color_9_submit_disc(env)
        board = arr(env.frame())
        observation.update(
            {
                "program": program,
                "level": env.levels_completed,
                "terminal": bool(env.terminal()),
                "cyan": board_components(board, 11),
                "white": board_components(board, 15),
                "board_delta": int((entry[:32, 32:] != board[:32, 32:]).sum()),
            }
        )

    levels, path, error = A.run_program("tn36", observe)
    observation.update({"result_levels": levels, "moves": len(path), "error": error})
    return observation


for value in sys.argv[1:]:
    print("fresh", run(value))

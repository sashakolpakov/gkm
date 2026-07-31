"""Run one level-6 direction program in a fresh Arena process."""

import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A

import solve
from legs import click_largest_color_9_submit_disc, find_right_segment_panel
from perception import arr, frame_delta


DIRECTION_ROWS = {
    "N": (),
    "D": (0, 1),
    "U": (0, 5),
    "R": (1,),
    "L": (0,),
    "M": (0, 3),
    "X": (3,),
    "A": (0, 2),
    "B": (0, 1, 2, 3, 4, 5),
}
PROGRAM = sys.argv[1]


def cyan_cells(frame):
    pixels = arr(frame)
    return tuple(
        (row_index, col_index, int((pixels[row : row + 4, col : col + 4] == 11).sum()))
        for row_index, row in enumerate(range(4, 32, 4))
        for col_index, col in enumerate(range(33, 61, 4))
        if int((pixels[row : row + 4, col : col + 4] == 11).sum())
    )


def write_program(env):
    frame = arr(env.frame()).copy()
    _, rows, cols = find_right_segment_panel(frame)
    for col, direction in zip(cols, PROGRAM):
        desired = DIRECTION_ROWS[direction]
        for row_index, row in enumerate(rows):
            if (int(frame[row][col]) == 5) != (row_index in desired):
                env.step(6, col, row)


def observe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)
    solve.solve(env)

    level = env.levels_completed
    entry = arr(env.frame()).copy()
    write_program(env)
    configured = arr(env.frame()).copy()
    click_largest_color_9_submit_disc(env)
    submitted = arr(env.frame()).copy()
    print(
        "one_program",
        {
            "program": PROGRAM,
            "entry_cyan": cyan_cells(entry),
            "configured_cyan": cyan_cells(configured),
            "submitted_cyan": cyan_cells(submitted),
            "config_delta": frame_delta(entry, configured),
            "submit_delta": frame_delta(configured, submitted),
            "level_delta": env.levels_completed - level,
        },
    )


levels, path, error = A.run_program("tn36", observe)
print("probe_result", {"levels": levels, "moves": len(path), "error": error})

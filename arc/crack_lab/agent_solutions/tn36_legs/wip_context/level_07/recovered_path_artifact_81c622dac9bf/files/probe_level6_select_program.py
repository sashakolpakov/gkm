"""Test hidden board selection followed by a six-symbol level-6 program."""

import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A

import solve
from legs import click_largest_color_9_submit_disc, find_right_segment_panel
from perception import arr, frame_delta


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
SELECT_X = int(sys.argv[1])
SELECT_Y = int(sys.argv[2])
PROGRAM = sys.argv[3]
ORDER = sys.argv[4] if len(sys.argv) > 4 else "before"


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
    for col, symbol in zip(cols, PROGRAM):
        for row_index in CODES[symbol]:
            env.step(6, col, rows[row_index])


def observe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)
    solve.solve(env)

    level = env.levels_completed
    entry = arr(env.frame()).copy()
    if ORDER == "before":
        env.step(6, SELECT_X, SELECT_Y)
    selected = arr(env.frame()).copy()
    write_program(env)
    configured = arr(env.frame()).copy()
    if ORDER == "after":
        env.step(6, SELECT_X, SELECT_Y)
    click_largest_color_9_submit_disc(env)
    submitted = arr(env.frame()).copy()
    print(
        "select_program",
        {
            "select": (SELECT_X, SELECT_Y),
            "order": ORDER,
            "program": PROGRAM,
            "select_delta": frame_delta(entry, selected),
            "configured_cyan": cyan_cells(configured),
            "submit_delta": frame_delta(configured, submitted),
            "submitted_cyan": cyan_cells(submitted),
            "level_delta": env.levels_completed - level,
            "terminal": bool(env.terminal()),
        },
    )


levels, path, error = A.run_program("tn36", observe)
print("probe_result", {"levels": levels, "moves": len(path), "error": error})

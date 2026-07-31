"""Trace upper-board state while constructing and submitting level-6 programs."""

import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A

import solve
from legs import click_largest_color_9_submit_disc, find_right_segment_panel
from perception import arr


DIRECTION_ROWS = {
    "D": (0, 1),
    "U": (0, 5),
    "R": (1,),
    "L": (0,),
}


def cyan_cells(frame):
    pixels = arr(frame)
    return tuple(
        (row_index, col_index, int((pixels[row : row + 4, col : col + 4] == 11).sum()))
        for row_index, row in enumerate(range(4, 32, 4))
        for col_index, col in enumerate(range(33, 61, 4))
        if int((pixels[row : row + 4, col : col + 4] == 11).sum())
    )


def clock(frame):
    pixels = arr(frame)
    return int((pixels[1, 1:62] == 3).sum())


def trace_program(root, program):
    clone = root.clone()
    initial = cyan_cells(clone.frame())
    snapshot = arr(clone.frame()).copy()
    _, rows, cols = find_right_segment_panel(snapshot)
    print("trace_start", {"program": program, "cyan": initial, "clock": clock(snapshot)})
    action_index = 0
    for col, direction in zip(cols, program):
        for row_index, row in enumerate(rows):
            if (int(snapshot[row][col]) == 5) != (row_index in DIRECTION_ROWS[direction]):
                clone.step(6, col, row)
                action_index += 1
                current = cyan_cells(clone.frame())
                print(
                    "trace_click",
                    {
                        "program": program,
                        "action": action_index,
                        "at": (col, row),
                        "cyan": current,
                        "changed": current != initial,
                        "clock": clock(clone.frame()),
                    },
                )
    before_submit = cyan_cells(clone.frame())
    click_largest_color_9_submit_disc(clone)
    print(
        "trace_submit",
        {
            "program": program,
            "before": before_submit,
            "after": cyan_cells(clone.frame()),
            "level_delta": clone.levels_completed - root.levels_completed,
            "clock": clock(clone.frame()),
        },
    )


def observe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)
    solve.solve(env)

    for program in ("RRRRRR", "LLLLLL", "RRRRUU"):
        trace_program(env, program)


levels, path, error = A.run_program("tn36", observe)
print("probe_result", {"levels": levels, "moves": len(path), "error": error})

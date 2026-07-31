"""Run a staged level-6 route chain in one fresh Arena environment."""

import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A

import solve
from legs import click_largest_color_9_submit_disc, find_right_segment_panel
from perception import arr, connected_components


DIRECTION_ROWS = {
    "D": (0, 1),
    "U": (0, 5),
    "R": (1,),
    "L": (0,),
    "M": (0, 3),
}
PROGRAMS = sys.argv[1:]


def cyan_cells(frame):
    pixels = arr(frame)
    return tuple(
        (row_index, col_index, int((pixels[row : row + 4, col : col + 4] == 11).sum()))
        for row_index, row in enumerate(range(4, 32, 4))
        for col_index, col in enumerate(range(33, 61, 4))
        if int((pixels[row : row + 4, col : col + 4] == 11).sum())
    )


def cyan_components(frame):
    return tuple(
        (blob.bbox, blob.area, blob.size)
        for blob in connected_components(frame, colors=(11,))
        if blob.bbox[2] < 32 and blob.bbox[1] > 32
    )


def write_program(env, program):
    frame = arr(env.frame()).copy()
    _, rows, cols = find_right_segment_panel(frame)
    for col, direction in zip(cols, program):
        desired = DIRECTION_ROWS[direction]
        for row_index, row in enumerate(rows):
            if (int(frame[row][col]) == 5) != (row_index in desired):
                env.step(6, col, row)


def summary(env):
    return {
        "level": env.levels_completed,
        "terminal": bool(env.terminal()),
        "cyan_cells": cyan_cells(env.frame()),
        "cyan_components": cyan_components(env.frame()),
    }


def observe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)
    solve.solve(env)

    print("chain_entry", summary(env))
    for index, program in enumerate(PROGRAMS):
        write_program(env, program)
        click_largest_color_9_submit_disc(env)
        print("chain_step", index, program, summary(env))
        if env.levels_completed >= 6:
            break


levels, path, error = A.run_program("tn36", observe)
print("probe_result", {"levels": levels, "moves": len(path), "error": error})

"""Small symbolic traces for level-8 collector/token interactions."""

import importlib.util
import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena
import numpy as np


spec = importlib.util.spec_from_file_location("solve", "solve.py")
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)

SELECT_TOP = (6, 37, 58)
SELECT_LEFT = (6, 14, 58)


def apply(env, action):
    if isinstance(action, tuple):
        env.step(*action)
    else:
        env.step(action)


def cell_symbol(cell):
    for color, symbol in ((8, "8"), (9, "9"), (12, "C"), (14, "E")):
        if np.any(cell == color):
            return symbol
    if np.any(cell == 6):
        return "H"
    if np.any(cell == 15):
        return "V"
    if any(np.any(cell == color) for color in (1, 2, 3)):
        return "+"
    return "."


def symbolic(frame):
    rows = []
    for grid_row in range(8):
        chars = []
        for grid_col in range(8):
            row = 2 + 6 * grid_row
            col = 5 + 6 * grid_col
            chars.append(cell_symbol(np.asarray(frame)[row : row + 6, col : col + 6]))
        rows.append("".join(chars))
    return "/".join(rows)


def positions(frame):
    pixels = np.asarray(frame)
    result = {}
    for color in (8, 9, 12, 14):
        points = np.argwhere(pixels[:53] == color)
        result[color] = tuple(round(float(value), 1) for value in points.mean(axis=0))
    return result


def run_trace(root, name, actions):
    clone = root.clone()
    print("TRACE", name, "0", symbolic(clone.frame()))
    for index, action in enumerate(actions, 1):
        apply(clone, action)
        print(
            "TRACE",
            name,
            index,
            action,
            clone.levels_completed,
            symbolic(clone.frame()),
            positions(clone.frame()),
        )


def probe(env):
    solver.solve(env)
    run_trace(
        env,
        "ATTACH_8_TO_TOP",
        [
            SELECT_TOP,
            2,
            3,
            SELECT_LEFT,
            4,
            2,
            4,
            1,
            1,
            SELECT_TOP,
            1,
            SELECT_LEFT,
            2,
            4,
            4,
            4,
            4,
            3,
            3,
            SELECT_TOP,
            4,
            4,
            SELECT_LEFT,
            1,
            1,
            SELECT_TOP,
            2,
            2,
            SELECT_LEFT,
            3,
            SELECT_TOP,
            1,
            3,
            2,
            2,
            2,
            2,
            2,
            3,
            1,
            1,
            1,
            SELECT_LEFT,
            3,
            3,
            SELECT_TOP,
            1,
            1,
            4,
            SELECT_LEFT,
            2,
            4,
            1,
            SELECT_TOP,
            3,
            SELECT_LEFT,
            3,
            3,
            3,
            1,
        ],
    )


levels, path, err = arena.run_program("sk48", probe)
print("TRACE_RESULT", levels, len(path), err)

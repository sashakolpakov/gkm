"""Measure each level-5 protocol glyph through the live board preview."""

import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A
import numpy as np

from perception import arr


ROWS = (33, 36, 39, 42, 45, 48)
COLS = (34, 39, 44, 49, 54, 59)
CODES = {
    "N": (),
    "R": (1,),
    "D": (0, 1),
    "U": (0, 5),
    "L": (1, 5),
    "X": (3,),
    "M": (0, 3),
    "A": (0, 2),
    "B": (0, 1, 2, 3, 4, 5),
}


def apply_program(node, program):
    for col, symbol in zip(COLS, program):
        for row_index in CODES[symbol]:
            node.step(6, col, ROWS[row_index])


def bbox(mask):
    rows, cols = np.where(mask)
    if not len(rows):
        return None
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def observe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)

    initial = arr(env.frame()).copy()
    board_slice = np.s_[4:32, 33:61]
    target = initial[board_slice] == 15
    source = initial[board_slice] == 11

    tests = (
        "NNNNNN",
        "RNNNNN",
        "DNNNNN",
        "UNNNNN",
        "LNNNNN",
        "XNNNNN",
        "MNNNNN",
        "ANNNNN",
        "BNNNNN",
        "DDNNNN",
        "DDDNNN",
        "RRRNNN",
        "LLLNNN",
        "UUUNNN",
        "XXNNNN",
        "XXXNNN",
        "MMNNNN",
        "MMMNNN",
        "AANNNN",
        "AAANNN",
        "BBNNNN",
        "BBBNNN",
        "DDDXXN",
        "DDDXXX",
        "DDDXXA",
        "DDDXXB",
        "XXDDDN",
        "AXXDDD",
    )
    for program in tests:
        clone = env.clone()
        apply_program(clone, program)
        frame = arr(clone.frame())
        board = frame[board_slice]
        cyan = board == 11
        print(
            program,
            {
                "cyan": int(cyan.sum()),
                "bbox": bbox(cyan),
                "target_covered": int((cyan & target).sum()),
                "cyan_off_target": int((cyan & ~target).sum()),
                "target_remaining": int((board == 15).sum()),
                "source_remaining": int((cyan & source).sum()),
                "board_changed": int((board != initial[board_slice]).sum()),
            },
        )


levels, path, error = A.run_program("tn36", observe)
print("probe_result", {"levels": levels, "moves": len(path), "error": error})

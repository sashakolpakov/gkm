"""Localize protocol-execution changes for representative level-5 programs."""

import json
import sys
from collections import Counter, defaultdict

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A

from perception import arr, frame_delta


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


def grouped_changes(before, after):
    groups = defaultdict(Counter)
    ys, xs = (before != after).nonzero()
    for row, col in zip(ys, xs):
        if row < 32 and col < 32:
            region = "demo"
        elif row < 32:
            region = "board"
        elif row < 52:
            region = "panels"
        else:
            region = "controls"
        groups[region][(int(before[row, col]), int(after[row, col]))] += 1
    return {region: tuple(sorted(counts.items())) for region, counts in groups.items()}


def board_lines(frame):
    symbols = {0: "0", 4: "Y", 5: ".", 6: "W", 11: "C", 15: "T"}
    return tuple(
        "".join(symbols.get(int(frame[row, col]), "?") for col in range(33, 61))
        for row in range(4, 32)
    )


def observe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)

    for program in ("DDDXXA", "DDDXXB", "DDDXXX", "DDDXXN", "DDDXXM", "DDDXAB"):
        clone = env.clone()
        apply_program(clone, program)
        before = arr(clone.frame()).copy()
        clone.step(6, 57, 58)
        after = arr(clone.frame()).copy()
        changed_rows = [
            (row + 4, old, new)
            for row, (old, new) in enumerate(zip(board_lines(before), board_lines(after)))
            if old != new
        ]
        print(
            program,
            {
                "delta": frame_delta(before, after),
                "regions": grouped_changes(before, after),
                "board_rows": changed_rows,
            },
        )


levels, path, error = A.run_program("tn36", observe)
print("probe_result", {"levels": levels, "moves": len(path), "error": error})

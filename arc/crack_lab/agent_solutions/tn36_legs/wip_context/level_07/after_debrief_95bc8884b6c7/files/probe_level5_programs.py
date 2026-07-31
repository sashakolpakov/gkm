"""Test a compact protocol vocabulary against the live level-5 board."""

import json
import sys
from collections import Counter

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A

from perception import arr, connected_components, frame_delta


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


def board_blobs(frame, color):
    return tuple(
        (blob.bbox, blob.area)
        for blob in connected_components(frame, colors=(color,))
        if blob.bbox[2] < 32 and blob.bbox[1] > 31
    )


def transition_counts(before, after):
    changed = arr(before) != arr(after)
    return tuple(
        sorted(
            Counter(
                zip(
                    (int(value) for value in arr(before)[changed]),
                    (int(value) for value in arr(after)[changed]),
                )
            ).items()
        )
    )


def observe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)

    programs = (
        "DDDXXA",
        "DDDXXB",
        "DDDXXX",
        "DDDXXN",
        "DDDXXM",
        "DDDAXB",
        "DDDXAB",
        "DDDAAX",
        "XXDDDA",
        "AXXDDD",
        "BDD DXX".replace(" ", ""),
        "DDDBXX",
        "DDDAMM",
        "DDDAAA",
        "DDDBBB",
    )
    for program in programs:
        clone = env.clone()
        apply_program(clone, program)
        before = arr(clone.frame()).copy()
        clone.step(6, 57, 58)
        after = arr(clone.frame()).copy()
        print(
            program,
            {
                "level_delta": clone.levels_completed - env.levels_completed,
                "cyan": board_blobs(after, 11),
                "target": board_blobs(after, 15),
                "submit_delta": frame_delta(before, after)["count"],
                "transitions": transition_counts(before, after),
            },
        )


levels, path, error = A.run_program("tn36", observe)
print("probe_result", {"levels": levels, "moves": len(path), "error": error})

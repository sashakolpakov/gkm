"""Read dense post-submit board feedback for isolated protocol glyphs."""

import json
import sys
from collections import Counter

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A

from perception import arr, connected_components


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
}


def apply_program(node, program):
    for col, symbol in zip(COLS, program):
        for row_index in CODES[symbol]:
            node.step(6, col, ROWS[row_index])


def normalized_shape(frame, blob):
    r0, c0, r1, c1 = blob.bbox
    return tuple(
        "".join("#" if int(frame[row, col]) == blob.color else "." for col in range(c0, c1 + 1))
        for row in range(r0, r1 + 1)
    )


def board_objects(frame):
    return tuple(
        (blob.bbox, blob.area, normalized_shape(frame, blob))
        for blob in connected_components(frame, colors=(11,))
        if blob.bbox[2] < 32 and blob.bbox[1] > 31
    )


def tile_counts(frame):
    return tuple(
        tuple(
            tuple(
                sorted(
                    Counter(
                        int(value)
                        for value in frame[row : row + 4, col : col + 4].flat
                    ).items()
                )
            )
            for col in range(33, 61, 4)
        )
        for row in range(4, 32, 4)
    )


def observe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)

    entry_tiles = tile_counts(arr(env.frame()))
    programs = tuple(sys.argv[1:]) or (
        "NNNNNN",
        "LNNNNN",
        "DNNNNN",
        "RNNNNN",
        "UNNNNN",
        "XNNNNN",
        "MNNNNN",
        "LLLLLL",
        "DDDDDD",
        "RRRRRR",
        "UUUUUU",
        "XXXXXX",
        "MMMMMM",
        "MXNNNN",
        "XMNNNN",
        "MLNNNN",
        "MDNNNN",
        "MLLDDD",
        "XLLDDD",
        "XMLDDD",
    )
    for program in programs:
        clone = env.clone()
        apply_program(clone, program)
        clone.step(6, 57, 58)
        frame = arr(clone.frame())
        current_tiles = tile_counts(frame)
        changed_tiles = tuple(
            (row, col, entry_tiles[row][col], current_tiles[row][col])
            for row in range(7)
            for col in range(7)
            if entry_tiles[row][col] != current_tiles[row][col]
        )
        print(
            program,
            {
                "level_delta": clone.levels_completed - env.levels_completed,
                "objects": board_objects(frame),
                "changed_tiles": changed_tiles,
            },
        )


levels, path, error = A.run_program("tn36", observe)
print("probe_result", {"levels": levels, "moves": len(path), "error": error})

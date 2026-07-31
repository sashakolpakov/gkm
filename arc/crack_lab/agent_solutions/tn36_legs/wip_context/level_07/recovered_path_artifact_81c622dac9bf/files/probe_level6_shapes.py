"""Extract exact compact masks for level-6 clues, agents, and sockets."""

import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A

import solve
from perception import arr, connected_components


def mask(frame, bbox, color):
    r0, c0, r1, c1 = bbox
    return tuple(
        "".join("#" if int(frame[row][col]) == color else "." for col in range(c0, c1 + 1))
        for row in range(r0, r1 + 1)
    )


def observe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)
    solve.solve(env)

    frame = arr(env.frame())
    print(
        "yellow",
        [
            (blob.bbox, blob.area, mask(frame, blob.bbox, 4))
            for blob in connected_components(frame, colors=(4,), min_area=4)
            if blob.bbox[2] < 32 and blob.bbox[1] < 32
        ],
    )
    print(
        "cyan_components",
        [
            (blob.bbox, blob.area, mask(frame, blob.bbox, 11))
            for blob in connected_components(frame, colors=(11,), min_area=4)
            if blob.bbox[2] < 32 and blob.bbox[1] > 32
        ],
    )
    print(
        "cyan_cells",
        [
            (
                (row_index, col_index),
                mask(frame, (row, col, row + 3, col + 3), 11),
            )
            for row_index, row in enumerate(range(4, 32, 4))
            for col_index, col in enumerate(range(33, 61, 4))
            if int((frame[row : row + 4, col : col + 4] == 11).sum())
        ],
    )


levels, path, error = A.run_program("tn36", observe)
print("probe_result", {"levels": levels, "moves": len(path), "error": error})

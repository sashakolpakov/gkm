"""Print compact masks and lattice cells for level-7 board objects."""

import json
import sys
from collections import Counter

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A

import solve
from perception import arr, connected_components


def mask(frame, blob):
    r0, c0, r1, c1 = blob.bbox
    return tuple(
        "".join("#" if int(frame[row][col]) == blob.color else "." for col in range(c0, c1 + 1))
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
        "objects",
        tuple(
            (blob.color, blob.bbox, blob.area, mask(frame, blob))
            for blob in connected_components(
                frame,
                colors=(11, 12, 13, 14),
                min_area=2,
            )
            if blob.bbox[2] < 32 and blob.bbox[1] >= 32
        ),
    )
    cells = []
    for row in range(4, 32, 4):
        line = []
        for col in range(33, 61, 4):
            counts = Counter(
                int(frame[r][c])
                for r in range(row, row + 4)
                for c in range(col, col + 4)
            )
            line.append(tuple(sorted(counts.items())))
        cells.append(tuple(line))
    print("lattice", tuple(cells))
    print(
        "sockets",
        {
            "left": tuple(
                "".join(format(int(frame[row][col]), "x") for col in range(37, 41))
                for row in range(16, 20)
            ),
            "right": tuple(
                "".join(format(int(frame[row][col]), "x") for col in range(57, 61))
                for row in range(16, 20)
            ),
        },
    )


levels, path, error = A.run_program("tn36", observe)
print("probe_result", {"levels": levels, "moves": len(path), "error": error})

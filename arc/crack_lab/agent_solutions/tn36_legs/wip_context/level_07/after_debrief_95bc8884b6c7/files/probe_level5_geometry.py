"""Exact symbolic geometry for the level-5 demonstration and live board."""

import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A

from perception import arr, connected_components


SYMBOLS = {0: "0", 1: "b", 2: "r", 3: "g", 4: "Y", 5: ".", 6: "W", 9: "9", 11: "C", 15: "T"}


def shape(frame, blob):
    r0, c0, r1, c1 = blob.bbox
    return tuple(
        "".join("#" if int(frame[row, col]) == blob.color else "." for col in range(c0, c1 + 1))
        for row in range(r0, r1 + 1)
    )


def observe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)
    frame = arr(env.frame())

    for name, color in (("cyan", 11), ("target", 15)):
        objects = [
            blob
            for blob in connected_components(frame, colors=(color,))
            if blob.bbox[2] < 32 and blob.bbox[1] > 31
        ]
        print(name, [(blob.bbox, blob.area, shape(frame, blob)) for blob in objects])

    print("board")
    for row in range(4, 32):
        print(f"{row:02}", "".join(SYMBOLS[int(frame[row, col])] for col in range(33, 61)))

    print("demonstrations")
    for name, col in (("max", 5), ("x", 15), ("down", 25), ("new_a", 35), ("new_b", 45)):
        clone = env.clone()
        clone.step(6, col, 58)
        current = arr(clone.frame())
        objects = [
            blob
            for blob in connected_components(current, colors=(4,))
            if blob.bbox[2] < 32 and blob.bbox[3] < 32
        ]
        print(name, [(blob.bbox, blob.area, shape(current, blob)) for blob in objects])

    print("buttons")
    for name, col in (("max", 5), ("x", 15), ("down", 25), ("new_a", 35), ("new_b", 45)):
        print(
            name,
            tuple(
                "".join(SYMBOLS[int(frame[row, column])] for column in range(col - 3, col + 4))
                for row in range(55, 62)
            ),
        )


levels, path, error = A.run_program("tn36", observe)
print("probe_result", {"levels": levels, "moves": len(path), "error": error})

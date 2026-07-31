"""Reproduce level-4 control effects in distinct pristine-clone contexts."""

import json
import sys
from collections import Counter

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A

from perception import arr, connected_components


ROWS = (33, 36, 39, 42, 45, 48)
LEFT_COLS = (11, 16, 21)
RIGHT_COLS = (34, 39, 44, 49, 54, 59)


def bits(frame, columns):
    return tuple(
        "".join("1" if int(frame[row, col]) == 5 else "0" for row in ROWS)
        for col in columns
    )


def normalized_shape(frame, blob):
    r0, c0, r1, c1 = blob.bbox
    return tuple(
        "".join("#" if int(frame[row, col]) == blob.color else "." for col in range(c0, c1 + 1))
        for row in range(r0, r1 + 1)
    )


def demo_objects(frame):
    return tuple(
        (blob.bbox, blob.area, normalized_shape(frame, blob))
        for blob in connected_components(frame, colors=(4,))
        if blob.bbox[1] < 31
    )


def board_objects(frame):
    return tuple(
        (blob.bbox, blob.area, normalized_shape(frame, blob))
        for blob in connected_components(frame, colors=(11,))
        if blob.bbox[2] < 32 and blob.bbox[1] > 31
    )


def board_tiles(frame):
    result = []
    for row in range(4, 32, 4):
        line = []
        for col in range(33, 61, 4):
            counts = Counter(int(value) for value in frame[row : row + 4, col : col + 4].flat)
            line.append("/".join(f"{color}:{count}" for color, count in sorted(counts.items())))
        result.append(tuple(line))
    return tuple(result)


def state(node):
    frame = arr(node.frame())
    selector = tuple(
        sum(int(value) == 9 for value in frame[54:63, col - 4 : col + 5].flat)
        for col in (5, 15, 25, 35)
    )
    return {
        "levels": node.levels_completed,
        "timer": sum(int(value) == 9 for value in frame[1, :]),
        "selector9": selector,
        "demo4": demo_objects(frame),
        "left_bits": bits(frame, LEFT_COLS),
        "right_bits": bits(frame, RIGHT_COLS),
        "board11": board_objects(frame),
    }


def observe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)

    base_tiles = board_tiles(arr(env.frame()))
    print("board_tiles")
    for line in base_tiles:
        print(" ".join(f"{cell:>9}" for cell in line))
    print("entry", state(env))

    tests = (
        ("selected_5", (5,)),
        ("control_15", (15,)),
        ("control_25", (25,)),
        ("control_35", (35,)),
        ("submit_57", (57,)),
        ("15_then_5", (15, 5)),
        ("25_then_5", (25, 5)),
        ("35_then_5", (35, 5)),
        ("15_then_25", (15, 25)),
        ("25_then_15", (25, 15)),
        ("15_twice", (15, 15)),
        ("25_twice", (25, 25)),
        ("35_twice", (35, 35)),
    )
    for name, columns in tests:
        clone = env.clone()
        for col in columns:
            clone.step(6, col, 58)
        current_tiles = board_tiles(arr(clone.frame()))
        changed_tiles = tuple(
            (row, col, base_tiles[row][col], current_tiles[row][col])
            for row in range(7)
            for col in range(7)
            if base_tiles[row][col] != current_tiles[row][col]
        )
        print(name, state(clone), "changed_tiles", changed_tiles)


levels, path, error = A.run_program("tn36", observe)
print("probe_result", {"levels": levels, "moves": len(path), "error": error})

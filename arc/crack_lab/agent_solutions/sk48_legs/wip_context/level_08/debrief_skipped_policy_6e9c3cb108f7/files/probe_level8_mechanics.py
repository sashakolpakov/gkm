"""Targeted symbolic probes for level-8 top-collector hooking."""

import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena
import numpy as np

from perception import connected_components


SELECT_TOP = (6, 37, 58)
SELECT_LEFT = (6, 14, 58)


def center(frame, color):
    blobs = [
        blob
        for blob in connected_components(frame, colors=(color,), min_area=16)
        if blob.centroid[0] < 53
    ]
    blob = min(blobs, key=lambda item: item.centroid[0])
    return tuple(round(value, 1) for value in blob.centroid)


def state(frame):
    pixels = np.asarray(frame)
    return tuple(center(frame, color) for color in (8, 9, 12, 14))


def symbolic(frame):
    pixels = np.asarray(frame)
    rows = []
    for grid_row in range(8):
        chars = []
        for grid_col in range(8):
            row = 2 + 6 * grid_row
            col = 5 + 6 * grid_col
            cell = pixels[row : row + 6, col : col + 6]
            symbol = "."
            for color, candidate in (
                (8, "8"),
                (9, "9"),
                (12, "C"),
                (14, "E"),
                (6, "H"),
                (15, "V"),
            ):
                if np.any(cell == color):
                    symbol = candidate
                    break
            if symbol == "." and any(
                np.any(cell == color) for color in (1, 2, 3)
            ):
                symbol = "+"
            chars.append(symbol)
        rows.append("".join(chars))
    return "/".join(rows)


def apply(env, action):
    if isinstance(action, tuple):
        env.step(*action)
    else:
        env.step(action)


def run(root, name, actions):
    node = root.clone()
    base_level = node.levels_completed
    print("TRACE", name, 0, state(node.frame()))
    for index, action in enumerate(actions, 1):
        apply(node, action)
        if node.levels_completed > base_level:
            print("WIN", name, index, action, node.levels_completed)
            return
        print("TRACE", name, index, action, state(node.frame()))
    print("FINAL", name, node.levels_completed, symbolic(node.frame()))


def probe(env):
    with open("checkpoint.json") as checkpoint_file:
        checkpoint = json.load(checkpoint_file)
    for action in checkpoint["final_path"]:
        env.step(action)

    run(
        env,
        "RIGHT_SWEEP",
        [SELECT_TOP, 4, 2, 2, 2, 2, 2, 3, 1, 1, 1, 1, 1],
    )
    run(
        env,
        "LEFT_SWEEP",
        [SELECT_TOP, 3, 2, 2, 2, 2, 2, 4, 1, 1, 1, 1, 1],
    )
    run(
        env,
        "SEPARATE_AND_HOOK_14",
        [
            SELECT_TOP,
            4,
            2,
            3,
            2,
            2,
            2,
            2,
            1,
            1,
            1,
            1,
            1,
        ],
    )
    run(
        env,
        "SEPARATE_WALL_HOOK_14",
        [
            SELECT_TOP,
            4,
            2,
            3,
            2,
            2,
            2,
            2,
            2,
            1,
            1,
            1,
            1,
            1,
        ],
    )
    run(
        env,
        "BUILD_HORIZONTAL",
        [
            SELECT_TOP,
            4,
            2,
            3,
            2,
            2,
            2,
            2,
            2,
            1,
            1,
            SELECT_LEFT,
            1,
            1,
            4,
            4,
            4,
            4,
            4,
            4,
            3,
            3,
            3,
            3,
            3,
            3,
            2,
            2,
            2,
            4,
            4,
            4,
        ],
    )
    run(
        env,
        "STAGE_HORIZONTAL_ON_TETHER",
        [
            SELECT_TOP,
            4,
            2,
            3,
            1,
            3,
            2,
            1,
            4,
            2,
            2,
            2,
            2,
            2,
            2,
            1,
            1,
            4,
            SELECT_LEFT,
            1,
            4,
            4,
            4,
            3,
            3,
            3,
            2,
            2,
            4,
            4,
            4,
            4,
            3,
            3,
            3,
        ],
    )
    run(
        env,
        "HANDOFF_14_ON_ARRIVAL",
        [
            4,
            4,
            4,
            3,
            3,
            3,
            SELECT_TOP,
            4,
            2,
            3,
            1,
            3,
            2,
            1,
            SELECT_LEFT,
            1,
            4,
            4,
            4,
            4,
            4,
            4,
            3,
            3,
            3,
            3,
            3,
            3,
            2,
            2,
            4,
            4,
            SELECT_TOP,
            4,
            2,
            2,
            2,
            2,
            2,
            2,
            1,
            1,
            3,
            SELECT_LEFT,
            3,
            SELECT_TOP,
            3,
        ],
    )
    run(
        env,
        "DETACH_14_ON_SHELF",
        [
            4,
            4,
            4,
            4,
            3,
            3,
            3,
            3,
            SELECT_TOP,
            4,
            2,
            3,
            1,
            3,
            2,
            1,
            SELECT_LEFT,
            4,
            4,
            4,
            SELECT_TOP,
            4,
            2,
            2,
            2,
            2,
            2,
            2,
            1,
            1,
            1,
            1,
            4,
            SELECT_LEFT,
            3,
            3,
            3,
            2,
            4,
            4,
            4,
            3,
            3,
            3,
            1,
            4,
            4,
            4,
            4,
            4,
            3,
            2,
            4,
            3,
            3,
            3,
            3,
            SELECT_TOP,
            2,
            2,
            2,
            2,
            1,
            1,
            1,
            1,
            1,
            1,
            SELECT_LEFT,
            1,
            4,
            4,
            4,
            4,
            4,
            4,
            3,
            3,
            2,
            3,
            3,
            3,
            SELECT_TOP,
            2,
            2,
            2,
            2,
            2,
            2,
            1,
            1,
            1,
            1,
            1,
            SELECT_LEFT,
            1,
            1,
            4,
            4,
            4,
            4,
            4,
            4,
            3,
            3,
            3,
            3,
            3,
            3,
            2,
            2,
            4,
            4,
            4,
            4,
            4,
            4,
            3,
            3,
            3,
            3,
            3,
        ],
    )
    shelf_prefix = (
        [4] * 4
        + [3] * 4
        + [SELECT_TOP, 4, 2, 3, 1, 3, 2, 1, SELECT_LEFT]
        + [4] * 3
        + [SELECT_TOP, 4]
        + [2] * 6
        + [1] * 4
        + [4]
    )
    collision_release_suffix = (
        [SELECT_LEFT]
        + [3] * 3
        + [2]
        + [4] * 4
        + [3] * 4
        + [1]
        + [4] * 5
        + [3, 1]
        + [3] * 3
        + [SELECT_TOP]
        + [2] * 5
        + [1] * 6
        + [3] * 2
        + [2, 1]
        + [4] * 2
        + [SELECT_LEFT, 2]
        + [4] * 6
        + [3] * 2
        + [SELECT_TOP]
        + [2] * 2
        + [SELECT_LEFT, 1]
        + [3] * 3
        + [SELECT_TOP]
        + [1] * 2
        + [3] * 2
        + [2, 1]
        + [4] * 2
        + [2] * 6
        + [1] * 5
        + [SELECT_LEFT]
        + [4] * 6
        + [3] * 6
        + [2] * 2
        + [4] * 6
        + [3] * 5
    )
    run(
        env,
        "FULL_COLLISION_RELEASE",
        shelf_prefix + collision_release_suffix,
    )


levels, path, err = arena.run_program("sk48", probe)
print("MECHANICS_RESULT", levels, len(path), err)

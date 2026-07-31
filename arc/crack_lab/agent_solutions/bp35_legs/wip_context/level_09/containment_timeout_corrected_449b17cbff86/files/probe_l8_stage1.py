"""Enumerate effective actions after the verified two-band opening."""

import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A
import gkm_legs as G

from legs import (
    COL_ANCHORS,
    ROW_ANCHORS,
    _cell_shape,
    avatar_column,
    band_shift,
    click_action,
)
from perception import arr


if G._workspace_taint_reason(os.getcwd()):
    raise SystemExit("tainted workspace")


OPENING = [
    click_action(3, 0),
    click_action(3, 1),
    click_action(3, 2),
    (4,),
    (4,),
]


def symbol(frame, i, j):
    color, area = _cell_shape(frame, i, j)
    if color in (3, 5):
        return "#"
    if color == 10:
        return "."
    if color in (9, 11):
        return "A"
    if color == 15:
        return "H" if area >= 20 else "h"
    if color == 12:
        return "S" if area >= 20 else "s"
    if color == 14:
        return "Y"
    if color == 8:
        return "g"
    return "T"


def lattice(frame):
    return "/".join(
        "".join(symbol(frame, i, j) for j in range(8))
        for i in range(10)
    )


def target(frame):
    for row in range(63):
        for col in range(64):
            color = int(frame[row][col])
            if color not in (0, 3, 5, 8, 9, 10, 11, 12, 14, 15):
                return color, row, col
    return None


def material_delta(before, after):
    return sum(
        1
        for row in range(63)
        for col in range(64)
        if int(before[row][col]) != int(after[row][col])
    )


def run(node, route):
    height = 0
    for action in route:
        if node.terminal():
            break
        before = arr(node.frame()).copy()
        node.step(*action)
        if not node.terminal():
            height += band_shift(before, node.frame())
    return height


def probe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)
    opening_gain = run(env, OPENING)
    base = arr(env.frame()).copy()
    print(
        "ROOT",
        "gain",
        opening_gain,
        "col",
        avatar_column(base),
        "target",
        target(base),
        lattice(base),
    )

    for action in ((3,), (4,), (7,)):
        node = env.clone()
        before = arr(node.frame()).copy()
        node.step(*action)
        print(
            "KEY",
            action,
            "alive",
            not node.terminal(),
            "level",
            node.levels_completed,
            "col",
            None if node.terminal() else avatar_column(node.frame()),
            "shift",
            0 if node.terminal() else band_shift(before, node.frame()),
            "target",
            None if node.terminal() else target(node.frame()),
            "grid",
            "" if node.terminal() else lattice(node.frame()),
        )

    changed = []
    for i, y in enumerate(ROW_ANCHORS):
        for j, x in enumerate(COL_ANCHORS):
            node = env.clone()
            before = arr(node.frame()).copy()
            node.step(6, x, y)
            after = node.frame()
            material = material_delta(before, after)
            if material:
                changed.append(
                    (
                        (i, j),
                        _cell_shape(before, i, j),
                        material,
                        node.terminal(),
                        0 if node.terminal() else band_shift(before, after),
                        None if node.terminal() else lattice(after),
                    )
                )
    print("CLICKS", len(changed))
    for item in changed:
        print("CLICK", item)


if __name__ == "__main__":
    levels, path, err = A.run_program("bp35", probe)
    print("RESULT", levels, len(path), err)

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
    band_shift,
    click_action,
)
from perception import arr


if G._workspace_taint_reason(os.getcwd()):
    raise SystemExit("tainted workspace")


TAIL = [
    (4,), (4,), (4,), click_action(8, 4), (6, 3, 3),
    (4,), (6, 3, 3), (4,),
    (3,), (3,), click_action(8, 3), (6, 3, 3), (3,), (6, 3, 9),
    (3,), (4,), (4,), click_action(8, 2), (6, 3, 9),
    (3,), (3,), (6, 3, 15), (3,), (3,),
    (4,), click_action(7, 2), (6, 3, 21), (4,), (6, 3, 15),
    (6, 3, 27),
    (3,), (6, 3, 33), (4,), (6, 3, 33),
    (6, 3, 15), (3,), (6, 3, 45),
    (6, 3, 57), (4,), (6, 3, 51),
    (3,), (6, 3, 27), (4,), (6, 3, 57),
]


def avatar_cell(frame):
    ys, xs = (arr(frame) == 9).nonzero()
    if not len(ys):
        return None
    return (
        min(range(10), key=lambda i: abs(ROW_ANCHORS[i] - int(ys.mean()))),
        min(range(8), key=lambda j: abs(COL_ANCHORS[j] - int(xs.mean()))),
    )


def controls(frame):
    return [y for y in ROW_ANCHORS if int(frame[y][3]) == 8]


def execute(root, opening, suffix=()):
    node = root.clone()
    height = 0
    for action in [*opening, *TAIL, *suffix]:
        if node.terminal():
            break
        before = arr(node.frame()).copy()
        node.step(*action)
        height += band_shift(before, node.frame())
    return (
        node.levels_completed,
        node.terminal(),
        height,
        avatar_cell(node.frame()),
        controls(node.frame()),
    )


def probe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)
    supports = [
        (i, j)
        for i in range(10)
        for j in range(8)
        if _cell_shape(env.frame(), i, j)[0] in (12, 14)
    ]
    base_opening = [click_action(2, 2)]
    print(
        "BASE",
        execute(env, base_opening),
        execute(env, base_opening, [(3,)]),
        execute(env, base_opening, [(4,)]),
    )
    for support in supports:
        if support != (2, 2):
            opening = [click_action(2, 2), click_action(*support)]
            print(
                "EXTRA",
                support,
                execute(env, opening),
                execute(env, opening, [(3,)]),
                execute(env, opening, [(4,)]),
            )


levels, path, err = A.run_program("bp35", probe)
print("RESULT", levels, len(path), err)

import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A
import gkm_legs as G

from legs import (
    COL_ANCHORS, ROW_ANCHORS, _cell_shape, band_shift, click_action,
    run_actions,
)
from perception import arr


if G._workspace_taint_reason(os.getcwd()):
    raise SystemExit("tainted workspace")


STAGE = [
    (4,), (4,), (4,), click_action(8, 4), (6, 3, 3),
    (4,), (6, 3, 3), (4,),
    (3,), (3,), click_action(8, 3), (6, 3, 3), (3,), (6, 3, 9),
    (3,), (4,), (4,), click_action(8, 2), (6, 3, 9),
    (3,), (3,), (6, 3, 15), (3,), (3,),
    (4,), click_action(7, 2), (6, 3, 21), (4,), (6, 3, 15),
    click_action(4, 1), (6, 3, 15), (3,), (6, 3, 21),
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


def result(node):
    return (
        node.levels_completed, node.terminal(), avatar_cell(node.frame()),
        controls(node.frame()),
    )


def probe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)
    run_actions(env, STAGE)
    supports = [
        (i, j) for i in range(10) for j in range(8)
        if _cell_shape(env.frame(), i, j)[0] in (12, 14)
    ]
    base = arr(env.frame()).copy()
    print("STAGE", result(env), supports)
    for cell in supports:
        for move in (3, 4):
            route = [click_action(*cell), (6, 3, 51), (move,)]
            middle = env.clone()
            run_actions(middle, route)
            visible = controls(middle.frame())
            for y2 in visible:
                node = env.clone()
                run_actions(node, route + [(6, 3, y2)])
                gain = band_shift(base, node.frame())
                outcome = result(node)
                if outcome[0] > 6 or (
                    not outcome[1] and outcome[2]
                    and outcome[2][0] == 6
                ):
                    print("TEST", cell, move, y2, gain, outcome)


levels, path, err = A.run_program("bp35", probe)
print("RESULT", levels, len(path), err)

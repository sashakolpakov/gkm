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


GRAVITY = (6, 3, 3)
PREFIX = [
    (4,), (4,), (4,), click_action(8, 4), GRAVITY, (4,), GRAVITY, (4,),
    (3,), (3,), click_action(8, 3), GRAVITY, (3,), (6, 3, 9), (3,),
    (4,), (4,), click_action(8, 2), (6, 3, 9),
    (3,), (3,), (6, 3, 15), (3,), (3,),
    (4,), click_action(7, 2), (6, 3, 21), (4,), (6, 3, 15),
    (3,),
]
HAZARD = click_action(6, 2)


def gravity_action(frame):
    return next(
        ((6, 3, y) for y in ROW_ANCHORS if int(frame[y][3]) == 8),
        None,
    )


def avatar_cell(frame):
    ys, xs = (arr(frame) == 9).nonzero()
    if not len(ys):
        return None
    return (
        min(range(10), key=lambda i: abs(ROW_ANCHORS[i] - int(ys.mean()))),
        min(range(8), key=lambda j: abs(COL_ANCHORS[j] - int(xs.mean()))),
    )


def hazard_shapes(frame):
    return [
        (i, j, _cell_shape(frame, i, j))
        for i in range(10) for j in range(8)
        if _cell_shape(frame, i, j)[0] == 15
    ]


def execute(root, template):
    node = root.clone()
    actions = []
    for token in template:
        action = gravity_action(node.frame()) if token == "g" else token
        if action is None or node.terminal():
            break
        node.step(*action)
        actions.append(action)
    return node, actions


def probe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)
    run_actions(env, PREFIX)
    base = arr(env.frame()).copy()
    print("START", avatar_cell(base), hazard_shapes(base))
    routes = [
        [HAZARD], [HAZARD, HAZARD], [HAZARD, HAZARD, HAZARD],
        ["g", HAZARD], ["g", HAZARD, HAZARD],
        [HAZARD, "g", (4,), "g"],
        ["g", HAZARD, (4,), "g"],
        ["g", (4,), HAZARD, "g"],
        [(3,), HAZARD, "g", (3,), "g"],
        [(3,), "g", HAZARD, (4,), "g"],
    ]
    for route in routes:
        node, actions = execute(env, route)
        print(
            "TRY", actions, band_shift(base, node.frame()),
            node.levels_completed, node.terminal(), avatar_cell(node.frame()),
            hazard_shapes(node.frame()),
        )


levels, path, err = A.run_program("bp35", probe)
print("RESULT", levels, len(path), err)

import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A
import gkm_legs as G

from legs import (
    ROW_ANCHORS, _cell_shape, click_action, run_actions,
)
from perception import connected_components


if G._workspace_taint_reason(os.getcwd()):
    raise SystemExit("tainted workspace")


PREFIX = [
    (4,), (4,), (4,), click_action(8, 4), (6, 3, 3),
    (4,), (6, 3, 3), (4,),
    (3,), (3,), click_action(8, 3), (6, 3, 3), (3,), (6, 3, 9),
    (3,), (4,), (4,), click_action(8, 2), (6, 3, 9),
    (3,), (3,), (6, 3, 15), (3,), (3,),
    (4,), click_action(7, 2), (6, 3, 21), (4,), (6, 3, 15),
]


def avatar(frame):
    blobs = connected_components(frame, colors=(9,), min_area=2)
    return blobs[0].bbox if blobs else None


def controls(frame):
    return [y for y in ROW_ANCHORS if int(frame[y][3]) == 8]


def probe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)
    cells = [
        (i, j) for i in range(10) for j in range(8)
        if _cell_shape(env.frame(), i, j)[0] in (12, 14)
    ]
    for cell in cells:
        node = env.clone()
        run_actions(node, [click_action(*cell), *PREFIX])
        print(
            "STAGE", cell, node.levels_completed, node.terminal(),
            avatar(node.frame()), controls(node.frame()),
        )
    for cell in ((1, 4), (2, 2), (2, 4)):
        stage = env.clone()
        run_actions(stage, [click_action(*cell), *PREFIX])
        for y in controls(stage.frame()):
            for move in (None, 3, 4):
                node = env.clone()
                route = [click_action(*cell), *PREFIX, (6, 3, y)]
                if move is not None:
                    route.append((move,))
                run_actions(node, route)
                print(
                    "FOLLOW", cell, y, move, node.levels_completed,
                    node.terminal(), avatar(node.frame()),
                    controls(node.frame()),
                )


levels, path, err = A.run_program("bp35", probe)
print("RESULT", levels, len(path), err)

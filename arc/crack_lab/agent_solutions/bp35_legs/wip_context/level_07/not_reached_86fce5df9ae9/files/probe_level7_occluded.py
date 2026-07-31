import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A
import gkm_legs as G

from legs import (
    ROW_ANCHORS, _cell_shape, band_shift, click_action, run_actions,
)
from perception import connected_components


if G._workspace_taint_reason(os.getcwd()):
    raise SystemExit("tainted workspace")


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
    tests = [
        ("under", [(4,), (4,), (4,), click_action(6, 4), (3,)]),
        ("left", [(4,), (4,), (4,), (3,), click_action(6, 4)]),
        (
            "under_flip",
            [(4,), (4,), (4,), click_action(6, 4), (6, 3, 3), (3,)],
        ),
    ]
    for label, route in tests:
        node = env.clone()
        run_actions(node, route)
        print(
            label, node.levels_completed, node.terminal(),
            avatar(node.frame()), _cell_shape(node.frame(), 6, 4),
        )
    stage = [(4,), (4,), (4,), (3,), click_action(6, 4)]
    run_actions(env, stage)
    base = env.frame()
    print("STAGE", avatar(base), controls(base), _cell_shape(base, 6, 4))
    for y1 in controls(base):
        for move in (3, 4):
            middle = env.clone()
            run_actions(middle, [(6, 3, y1), (move,)])
            for y2 in controls(middle.frame()):
                node = env.clone()
                run_actions(node, [(6, 3, y1), (move,), (6, 3, y2)])
                if not node.terminal() and avatar(node.frame()):
                    print(
                        "PAIR", y1, move, y2,
                        band_shift(base, node.frame()), avatar(node.frame()),
                        controls(node.frame()),
                    )


levels, path, err = A.run_program("bp35", probe)
print("RESULT", levels, len(path), err)

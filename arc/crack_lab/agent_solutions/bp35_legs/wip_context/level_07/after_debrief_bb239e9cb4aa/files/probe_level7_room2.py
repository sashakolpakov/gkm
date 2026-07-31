import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A
import gkm_legs as G

from perception import connected_components
from legs import (
    COL_ANCHORS, ROW_ANCHORS, band_shift, click_action, run_actions,
)


if G._workspace_taint_reason(os.getcwd()):
    raise SystemExit("tainted workspace")


def avatar(frame):
    blobs = connected_components(frame, colors=(9,), min_area=2)
    return blobs[0].bbox if blobs else None


def probe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)
    opener = [
        (4,), (4,), (4,), click_action(8, 4),
        (6, 3, 3), (4,), (6, 3, 3), (4,),
    ]
    run_actions(env, opener)
    before = env.frame()
    supports = [
        (i, j) for i, y in enumerate(ROW_ANCHORS)
        for j, x in enumerate(COL_ANCHORS)
        if int(before[y][x]) == 12
    ]
    print("SUPPORTS", supports)
    for cell in supports:
        for suffix in ([(6, 3, 3)], [(4,)], [(3,)], [(6, 3, 3), (3,)]):
            node = env.clone()
            route = [click_action(*cell)] + suffix
            run_actions(node, route)
            print(cell, route[1:], node.terminal(), node.levels_completed,
                  avatar(node.frame()), band_shift(before, node.frame()))


levels, path, err = A.run_program("bp35", probe)
print("RESULT", levels, len(path), err)

"""Route the propagated support around the target wall via the right edge."""

import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A
import gkm_legs as G

from legs import _cell_shape, avatar_column, band_shift, click_action
from perception import arr, connected_components
from probe_l8_overwall import OVERWALL
from probe_l8_stage1 import lattice, target


if G._workspace_taint_reason(os.getcwd()):
    raise SystemExit("tainted workspace")


SAFE_LEFT = [(3,)] * 6
EDGE_ROUTE = [
    click_action(5, 7),
    (6, 63, 33),
    (6, 63, 39),
    (6, 63, 45),
    (6, 63, 51),
]


def edge_blobs(frame):
    return tuple(
        (blob.bbox, blob.area)
        for blob in connected_components(frame, colors=(15,), min_area=2)
        if blob.bbox[1] >= 55 and blob.bbox[0] < 63
    )


def run(node, route, trace=False):
    height = 0
    for index, action in enumerate(route, 1):
        if node.terminal():
            break
        before = arr(node.frame()).copy()
        shape = None
        if len(action) == 3 and 15 <= action[1] <= 57:
            shape = _cell_shape(
                before, (action[2] - 3) // 6, (action[1] - 15) // 6
            )
        node.step(*action)
        shift = 0 if node.terminal() else band_shift(before, node.frame())
        height += shift
        if trace:
            print(
                "STEP",
                index,
                action,
                "shape",
                shape,
                "alive",
                not node.terminal(),
                "level",
                node.levels_completed,
                "height",
                height,
                "col",
                None if node.terminal() else avatar_column(node.frame()),
                "target",
                None if node.terminal() else target(node.frame()),
                "edge",
                () if node.terminal() else edge_blobs(node.frame()),
                "grid",
                "" if node.terminal() else lattice(node.frame()),
            )
    return height


def probe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)

    root_route = [*OVERWALL, *SAFE_LEFT]
    for count in range(len(EDGE_ROUTE) + 1):
        node = env.clone()
        gain = run(node, [*root_route, *EDGE_ROUTE[:count]])
        print(
            "CASE",
            count,
            "alive",
            not node.terminal(),
            "level",
            node.levels_completed,
            "height",
            gain,
            "col",
            None if node.terminal() else avatar_column(node.frame()),
            "target",
            None if node.terminal() else target(node.frame()),
            "edge",
            () if node.terminal() else edge_blobs(node.frame()),
            "grid",
            "" if node.terminal() else lattice(node.frame()),
        )

    print("DIRECT")
    run(env, [*root_route, *EDGE_ROUTE], trace=True)


levels, path, err = A.run_program("bp35", probe)
print("RESULT", levels, len(path), err)

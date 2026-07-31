"""Audit small color-0/11 components for autonomous agents."""

import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A
import gkm_legs as G

from perception import connected_components, frame_delta


if G._workspace_taint_reason(os.getcwd()):
    raise SystemExit("tainted workspace")


def components(frame):
    return tuple(
        (blob.color, blob.bbox, blob.area)
        for blob in connected_components(frame, colors=(0, 9, 11, 14), min_area=1)
        if blob.bbox[0] < 63
    )


def probe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)

    base = env.frame()
    print("ROOT", components(base))
    routes = {
        "left": [(3,)],
        "right": [(4,)],
        "undo": [(7,)],
        "click": [(6, 15, 21)],
        "right_undo": [(4,), (7,)],
    }
    for label, route in routes.items():
        node = env.clone()
        for action in route:
            node.step(*action)
        print(
            "CASE",
            label,
            route,
            node.levels_completed,
            node.terminal(),
            frame_delta(base, node.frame())["bbox"],
            components(node.frame()),
        )


levels, path, err = A.run_program("bp35", probe)
print("RESULT", levels, len(path), err)

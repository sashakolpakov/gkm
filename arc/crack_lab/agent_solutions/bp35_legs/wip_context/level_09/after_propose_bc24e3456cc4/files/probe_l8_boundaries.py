"""Test horizontal boundaries at the target's world row."""

import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A
import gkm_legs as G

from perception import connected_components
from probe_l8_climb4 import ROOT_ROUTE
from probe_l8_stage1 import lattice, target


if G._workspace_taint_reason(os.getcwd()):
    raise SystemExit("tainted workspace")


def avatar(frame):
    blobs = connected_components(frame, colors=(9,), min_area=2)
    if not blobs:
        return None
    blob = blobs[0]
    return blob.bbox, blob.centroid


def run(node, route):
    for action in route:
        if node.terminal():
            break
        node.step(*action)


def probe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)

    root = env.clone()
    run(root, ROOT_ROUTE)
    print("ROOT", root.levels_completed, avatar(root.frame()), target(root.frame()), lattice(root.frame()))
    for direction in (3, 4):
        for count in range(1, 11):
            node = root.clone()
            run(node, [(direction,)] * count)
            print(
                "WALK",
                direction,
                count,
                node.levels_completed,
                node.terminal(),
                None if node.terminal() else avatar(node.frame()),
                None if node.terminal() else target(node.frame()),
                "" if node.terminal() else lattice(node.frame()),
            )


levels, path, err = A.run_program("bp35", probe)
print("RESULT", levels, len(path), err)

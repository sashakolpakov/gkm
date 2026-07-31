"""Test entry and target interactions at the pocket's color-12 opening."""

import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A
import gkm_legs as G

from legs import avatar_column, band_shift, click_action
from perception import arr, connected_components
from probe_l8_overwall import OVERWALL
from probe_l8_stage1 import lattice, target


if G._workspace_taint_reason(os.getcwd()):
    raise SystemExit("tainted workspace")


SHRINK = click_action(7, 6)


def target_click(frame):
    blobs = [
        blob
        for blob in connected_components(frame, colors=(7,), min_area=2)
        if blob.bbox[0] < 63
    ]
    if not blobs:
        return None
    blob = blobs[0]
    return 6, round(blob.centroid[1]), round(blob.centroid[0])


def run(node, route):
    signed_frames = []
    for action in route:
        if node.terminal():
            break
        before = arr(node.frame()).copy()
        node.step(*action)
        shift = 0 if node.terminal() else band_shift(before, node.frame())
        signed_frames.append((action, shift))
    return signed_frames


def summary(node):
    return (
        node.levels_completed,
        node.terminal(),
        None if node.terminal() else avatar_column(node.frame()),
        None if node.terminal() else target(node.frame()),
        "" if node.terminal() else lattice(node.frame()),
    )


def probe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)

    root = env.clone()
    run(root, OVERWALL)
    click = target_click(root.frame())
    variants = {
        "shrink_left": [SHRINK, (3,)],
        "shrink_left_target": [SHRINK, (3,), click],
        "shrink_left_left": [SHRINK, (3,), (3,)],
        "shrink_left_right": [SHRINK, (3,), (4,)],
        "shrink_left_empty": [SHRINK, (3,), (6, 15, 3)],
        "shrink_left_undo": [SHRINK, (3,), (7,)],
        "shrink_target": [SHRINK, click],
    }
    for label, route in variants.items():
        node = root.clone()
        events = run(node, route)
        print("CASE", label, events, summary(node))


levels, path, err = A.run_program("bp35", probe)
print("RESULT", levels, len(path), err)

"""Test the upper-right climb after crossing above the target wall."""

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


RELEASE = click_action(5, 7)


def controls(frame):
    return tuple(
        (blob.bbox, blob.area)
        for blob in connected_components(frame, colors=(8,), min_area=1)
        if blob.bbox[0] < 63
    )


def run(node, route):
    height = 0
    for action in route:
        if node.terminal():
            break
        before = arr(node.frame()).copy()
        node.step(*action)
        if not node.terminal():
            height += band_shift(before, node.frame())
    return height


def probe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)

    for count in range(1, 13):
        node = env.clone()
        gain = run(node, [*OVERWALL, *([RELEASE] * count)])
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
            "controls",
            () if node.terminal() else controls(node.frame()),
            "grid",
            "" if node.terminal() else lattice(node.frame()),
        )


levels, path, err = A.run_program("bp35", probe)
print("RESULT", levels, len(path), err)

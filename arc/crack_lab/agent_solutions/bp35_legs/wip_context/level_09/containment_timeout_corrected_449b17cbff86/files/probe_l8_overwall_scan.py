"""Scan every cell interaction at the target-pocket over-wall landing."""

import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A
import gkm_legs as G

from legs import (
    COL_ANCHORS,
    ROW_ANCHORS,
    _cell_shape,
    avatar_column,
    band_shift,
    click_action,
)
from perception import arr
from probe_l8_overwall import OVERWALL
from probe_l8_stage1 import lattice, material_delta, target


if G._workspace_taint_reason(os.getcwd()):
    raise SystemExit("tainted workspace")


def run(node, route):
    for action in route:
        if node.terminal():
            break
        node.step(*action)


def scan(label, root):
    print(
        "ROOT",
        label,
        root.levels_completed,
        root.terminal(),
        avatar_column(root.frame()),
        target(root.frame()),
        lattice(root.frame()),
    )
    for i, y in enumerate(ROW_ANCHORS):
        for j, x in enumerate(COL_ANCHORS):
            node = root.clone()
            before = arr(node.frame()).copy()
            node.step(6, x, y)
            material = material_delta(before, node.frame())
            if not material:
                continue
            print(
                "CLICK",
                label,
                (i, j),
                _cell_shape(before, i, j),
                material,
                node.levels_completed,
                node.terminal(),
                0 if node.terminal() else band_shift(before, node.frame()),
                None if node.terminal() else avatar_column(node.frame()),
                None if node.terminal() else target(node.frame()),
                "" if node.terminal() else lattice(node.frame()),
            )


def probe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)

    over = env.clone()
    run(over, OVERWALL)
    scan("over", over)

    opened = over.clone()
    run(opened, [click_action(7, 6), (3,)])
    scan("opened_c6", opened)


levels, path, err = A.run_program("bp35", probe)
print("RESULT", levels, len(path), err)

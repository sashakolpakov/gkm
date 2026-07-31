"""Probe the upper maze and its nearby small hazards."""

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
from probe_l8_climb4 import RELEASE, ROOT_ROUTE
from probe_l8_stage1 import lattice, material_delta, target


if G._workspace_taint_reason(os.getcwd()):
    raise SystemExit("tainted workspace")


TOP_ROUTE = [*ROOT_ROUTE, *([RELEASE] * 9)]


def run(node, route):
    gain = 0
    for action in route:
        if node.terminal():
            break
        before = arr(node.frame()).copy()
        node.step(*action)
        if not node.terminal():
            gain += band_shift(before, node.frame())
    return gain


def summary(node, gain):
    return (
        node.levels_completed,
        node.terminal(),
        gain,
        None if node.terminal() else avatar_column(node.frame()),
        None if node.terminal() else target(node.frame()),
        "" if node.terminal() else lattice(node.frame()),
    )


def probe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)

    top = env.clone()
    base_gain = run(top, TOP_ROUTE)
    print("ROOT", summary(top, base_gain))

    actions = []
    for i, y in enumerate(ROW_ANCHORS):
        for j, x in enumerate(COL_ANCHORS):
            node = top.clone()
            before = arr(node.frame()).copy()
            node.step(6, x, y)
            material = material_delta(before, node.frame())
            if not material:
                continue
            actions.append((6, x, y))
            gain = base_gain
            if not node.terminal():
                gain += band_shift(before, node.frame())
            print(
                "CLICK",
                (i, j),
                _cell_shape(before, i, j),
                material,
                summary(node, gain),
            )
            if node.terminal():
                continue
            child = node.clone()
            before_release = arr(child.frame()).copy()
            child.step(*RELEASE)
            child_gain = gain
            if not child.terminal():
                child_gain += band_shift(before_release, child.frame())
            print("THEN_RELEASE", (i, j), summary(child, child_gain))

    variants = {
        "hazard_then_release": [click_action(4, 1), RELEASE],
        "handoff_c2": [click_action(6, 2), (4,)],
        "handoff_c2_release": [click_action(6, 2), (4,), click_action(5, 2)],
        "handoff_c3": [
            click_action(6, 2),
            (4,),
            click_action(6, 3),
            (4,),
        ],
    }
    for label, suffix in variants.items():
        node = env.clone()
        gain = run(node, [*TOP_ROUTE, *suffix])
        print("CASE", label, suffix, summary(node, gain))


if __name__ == "__main__":
    levels, path, err = A.run_program("bp35", probe)
    print("RESULT", levels, len(path), err)

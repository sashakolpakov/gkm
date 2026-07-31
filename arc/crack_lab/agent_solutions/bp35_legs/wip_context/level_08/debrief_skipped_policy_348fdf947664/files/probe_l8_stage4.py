"""Enumerate exits from the first target-visible landing."""

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
from probe_l8_climb2 import RELEASE, ROOT_ROUTE
from probe_l8_stage1 import lattice, material_delta, target


if G._workspace_taint_reason(os.getcwd()):
    raise SystemExit("tainted workspace")


PREFIX = [*ROOT_ROUTE, *([RELEASE] * 4)]


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


def describe(node, gain):
    return (
        not node.terminal(),
        node.levels_completed,
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
    base_gain = run(env, PREFIX)
    print("ROOT", describe(env, base_gain))

    frame = env.frame()
    actions = [(3,), (4,), (7,)]
    for i, y in enumerate(ROW_ANCHORS):
        for j, x in enumerate(COL_ANCHORS):
            color, area = _cell_shape(frame, i, j)
            if color in (8, 12, 14, 15):
                actions.append((6, x, y))

    for action in actions:
        node = env.clone()
        before = arr(node.frame()).copy()
        shape = (
            None
            if len(action) == 1
            else _cell_shape(
                before,
                (action[2] - ROW_ANCHORS[0]) // 6,
                (action[1] - COL_ANCHORS[0]) // 6,
            )
        )
        node.step(*action)
        material = material_delta(before, node.frame())
        if len(action) == 3 and material == 0:
            continue
        gain = base_gain
        if not node.terminal():
            gain += band_shift(before, node.frame())
        state = describe(node, gain)
        print("ONE", action, shape, "delta", material, state)
        if node.terminal():
            continue
        for move in ((3,), (4,)):
            child = node.clone()
            before_move = arr(child.frame()).copy()
            child.step(*move)
            child_gain = gain
            if not child.terminal():
                child_gain += band_shift(before_move, child.frame())
            print("THEN", action, move, describe(child, child_gain))


if __name__ == "__main__":
    levels, path, err = A.run_program("bp35", probe)
    print("RESULT", levels, len(path), err)

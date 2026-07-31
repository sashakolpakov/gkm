"""Enumerate the hazard interaction needed after the first release."""

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
from probe_l8_stage1 import OPENING, lattice, material_delta, target


if G._workspace_taint_reason(os.getcwd()):
    raise SystemExit("tainted workspace")


PREFIX = [*OPENING, click_action(5, 3)]
RELEASE = click_action(5, 3)


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


def result(node, gain):
    return (
        "alive",
        not node.terminal(),
        "level",
        node.levels_completed,
        "gain",
        gain,
        "col",
        None if node.terminal() else avatar_column(node.frame()),
        "target",
        None if node.terminal() else target(node.frame()),
        "grid",
        "" if node.terminal() else lattice(node.frame()),
    )


def probe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)
    prefix_gain = run(env, PREFIX)
    print("ROOT", result(env, prefix_gain))

    actions = [(3,), (4,), (7,)]
    for i, y in enumerate(ROW_ANCHORS):
        for j, x in enumerate(COL_ANCHORS):
            node = env.clone()
            before = arr(node.frame()).copy()
            node.step(6, x, y)
            if material_delta(before, node.frame()):
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
        gain = prefix_gain
        node.step(*action)
        if not node.terminal():
            gain += band_shift(before, node.frame())
        print("ONE", action, "shape", shape, result(node, gain))
        if node.terminal():
            continue
        before_release = arr(node.frame()).copy()
        node.step(*RELEASE)
        if not node.terminal():
            gain += band_shift(before_release, node.frame())
        print("THEN_RELEASE", action, result(node, gain))


levels, path, err = A.run_program("bp35", probe)
print("RESULT", levels, len(path), err)

"""Verify support propagation as a safe opening for level 8."""

import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A
import gkm_legs as G

from legs import ROW_ANCHORS, avatar_column, band_shift, click_action
from perception import arr


if G._workspace_taint_reason(os.getcwd()):
    raise SystemExit("tainted workspace")


def full_cells(frame):
    cells = []
    for i, y in enumerate(ROW_ANCHORS):
        for j in range(8):
            color = int(frame[y][15 + 6 * j])
            if color not in (10, 3, 5, 9, 11):
                cells.append((i, j, color))
    return tuple(cells)


def run(node, route, trace=False):
    gained = 0
    for index, action in enumerate(route, 1):
        if node.terminal():
            break
        before = arr(node.frame()).copy()
        node.step(*action)
        shift = 0 if node.terminal() else band_shift(before, node.frame())
        gained += shift
        if trace:
            print(
                "STEP",
                index,
                action,
                "alive",
                not node.terminal(),
                "level",
                node.levels_completed,
                "col",
                None if node.terminal() else avatar_column(node.frame()),
                "gain",
                gained,
                "cells",
                () if node.terminal() else full_cells(node.frame()),
            )
    return gained


def probe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)

    rights = [(4,), (4,)]
    variants = {}
    for count in range(1, 6):
        propagate = [click_action(3, j) for j in range(count)]
        variants[f"horizontal_{count}"] = [*propagate, *rights]
    variants["upper_then_right"] = [
        click_action(3, 0),
        click_action(2, 0),
        click_action(2, 1),
        click_action(2, 2),
        click_action(2, 3),
        *rights,
    ]
    variants["lower_then_right"] = [
        click_action(3, 0),
        click_action(4, 0),
        click_action(4, 1),
        click_action(4, 2),
        click_action(4, 3),
        *rights,
    ]
    for label, route in variants.items():
        node = env.clone()
        gain = run(node, route)
        print(
            "CASE",
            label,
            len(route),
            "alive",
            not node.terminal(),
            "level",
            node.levels_completed,
            "col",
            None if node.terminal() else avatar_column(node.frame()),
            "gain",
            gain,
            "cells",
            () if node.terminal() else full_cells(node.frame()),
        )

    direct_route = variants["horizontal_3"]
    print("DIRECT", direct_route)
    run(env, direct_route, trace=True)


levels, path, err = A.run_program("bp35", probe)
print("RESULT", levels, len(path), err)

"""Test repeated release of the support directly above the avatar."""

import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A
import gkm_legs as G

from legs import _cell_shape, avatar_column, band_shift, click_action
from perception import arr
from probe_l8_stage1 import OPENING, lattice, target


if G._workspace_taint_reason(os.getcwd()):
    raise SystemExit("tainted workspace")


RELEASE = click_action(5, 3)


def run(node, route, trace=False):
    height = 0
    for index, action in enumerate(route, 1):
        if node.terminal():
            break
        before = arr(node.frame()).copy()
        pre = _cell_shape(before, 5, 3)
        node.step(*action)
        shift = 0 if node.terminal() else band_shift(before, node.frame())
        height += shift
        if trace:
            print(
                "STEP",
                index,
                action,
                "pre",
                pre,
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
                "grid",
                "" if node.terminal() else lattice(node.frame()),
            )
    return height


def probe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)

    for count in range(1, 13):
        node = env.clone()
        height = run(node, [*OPENING, *([RELEASE] * count)])
        print(
            "CASE",
            count,
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
            "grid",
            "" if node.terminal() else lattice(node.frame()),
        )

    print("DIRECT")
    run(env, [*OPENING, RELEASE, RELEASE], trace=True)


levels, path, err = A.run_program("bp35", probe)
print("RESULT", levels, len(path), err)

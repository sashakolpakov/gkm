"""Scan the raw pixels of the pocket's shrunken color-12 support."""

import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A
import gkm_legs as G

from legs import _cell_shape, avatar_column, band_shift, click_action
from perception import arr, frame_delta
from probe_l8_overwall import OVERWALL
from probe_l8_stage1 import lattice, material_delta, target


if G._workspace_taint_reason(os.getcwd()):
    raise SystemExit("tainted workspace")


SAFE_LEFT = [(3,)] * 6
SHRINK = click_action(7, 6)


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
    run(root, [*OVERWALL, *SAFE_LEFT, SHRINK])
    base = arr(root.frame()).copy()
    print(
        "ROOT",
        _cell_shape(base, 7, 6),
        avatar_column(base),
        target(base),
        lattice(base),
    )

    outcomes = {}
    for action_kind in (6, 7):
        for y in range(42, 48):
            for x in range(49, 55):
                node = root.clone()
                before = arr(node.frame()).copy()
                node.step(action_kind, x, y)
                material = material_delta(before, node.frame())
                if material == 0:
                    continue
                outcome = (
                    node.levels_completed,
                    node.terminal(),
                    None if node.terminal() else _cell_shape(node.frame(), 7, 6),
                    None if node.terminal() else avatar_column(node.frame()),
                    None if node.terminal() else target(node.frame()),
                    frame_delta(before, node.frame())["bbox"],
                    "" if node.terminal() else lattice(node.frame()),
                )
                outcomes.setdefault((action_kind, outcome), []).append((x, y))
    for (action_kind, outcome), points in outcomes.items():
        print("OUTCOME", action_kind, points, outcome)


levels, path, err = A.run_program("bp35", probe)
print("RESULT", levels, len(path), err)

"""Trace the central upper-maze shafts after the safe handoff."""

import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A
import gkm_legs as G

from legs import avatar_column, band_shift, click_action
from perception import arr
from probe_l8_stage1 import lattice, target
from probe_l8_top import TOP_ROUTE


if G._workspace_taint_reason(os.getcwd()):
    raise SystemExit("tainted workspace")


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


def probe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)

    for column in (2, 3, 4):
        handoff = []
        for next_col in range(2, column + 1):
            handoff.extend([click_action(6, next_col), (4,)])
        release = click_action(5, column)
        for count in range(1, 6):
            node = env.clone()
            gain = run(node, [*TOP_ROUTE, *handoff, *([release] * count)])
            print(
                "CASE",
                column,
                count,
                node.levels_completed,
                node.terminal(),
                gain,
                None if node.terminal() else avatar_column(node.frame()),
                None if node.terminal() else target(node.frame()),
                "" if node.terminal() else lattice(node.frame()),
            )


levels, path, err = A.run_program("bp35", probe)
print("RESULT", levels, len(path), err)

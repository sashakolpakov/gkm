"""Verify the first downward handoff after the top gravity flip."""

import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A
import gkm_legs as G

from legs import _cell_shape, band_shift, click_action
from perception import arr, connected_components
from probe_l8_flip import FLIP, route_to_column
from probe_l8_stage1 import lattice, target
from probe_l8_top import TOP_ROUTE


if G._workspace_taint_reason(os.getcwd()):
    raise SystemExit("tainted workspace")


FLIPPED_ROUTE = [*TOP_ROUTE, *route_to_column(3), FLIP]
HANDOFF = [click_action(4, 4), (4,)]
RELEASE = click_action(5, 4)


def avatar_cell(frame):
    blobs = connected_components(frame, colors=(9,), min_area=2)
    if not blobs:
        return None
    blob = blobs[0]
    row = min(range(10), key=lambda i: abs(3 + 6 * i - blob.centroid[0]))
    col = min(range(8), key=lambda j: abs(15 + 6 * j - blob.centroid[1]))
    return row, col


def run(node, route):
    for action in route:
        if node.terminal():
            break
        node.step(*action)


def summary(node):
    return (
        node.levels_completed,
        node.terminal(),
        None if node.terminal() else avatar_cell(node.frame()),
        None if node.terminal() else target(node.frame()),
        "" if node.terminal() else lattice(node.frame()),
    )


def probe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)

    flipped = env.clone()
    run(flipped, FLIPPED_ROUTE)
    print("FLIPPED", summary(flipped))

    for action in ((3,), (4,), click_action(4, 2), click_action(4, 4)):
        node = flipped.clone()
        before = arr(node.frame()).copy()
        node.step(*action)
        print(
            "ONE",
            action,
            0 if node.terminal() else band_shift(before, node.frame()),
            0 if node.terminal() else band_shift(node.frame(), before),
            summary(node),
        )

    for count in range(1, 13):
        node = env.clone()
        run(node, [*FLIPPED_ROUTE, *HANDOFF, *([RELEASE] * count)])
        print("CASE", count, summary(node))


if __name__ == "__main__":
    levels, path, err = A.run_program("bp35", probe)
    print("RESULT", levels, len(path), err)

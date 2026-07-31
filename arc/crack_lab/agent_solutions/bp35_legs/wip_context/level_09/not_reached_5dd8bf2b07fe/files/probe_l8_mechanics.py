"""Compact, bounded mechanics probes from the pristine level-8 entry."""

import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A
import gkm_legs as G

from legs import COL_ANCHORS, ROW_ANCHORS, avatar_column
from perception import arr, color_counts, connected_components, frame_delta


if G._workspace_taint_reason(os.getcwd()):
    raise SystemExit("tainted workspace")


PALETTE = {
    3: "#",
    5: "#",
    7: "T",
    8: "g",
    9: "A",
    10: ".",
    11: "a",
    12: "s",
    14: "Y",
    15: "h",
}


def lattice(frame):
    return "/".join(
        "".join(PALETTE.get(int(frame[y][x]), "?") for x in COL_ANCHORS)
        for y in ROW_ANCHORS
    )


def objects(frame):
    return [
        (blob.color, blob.bbox, blob.area)
        for blob in connected_components(
            frame, colors=(7, 8, 9, 11, 12, 14, 15), min_area=2
        )
    ]


def summary(node, root_frame):
    frame = node.frame()
    delta = frame_delta(root_frame, frame)
    return (
        "dead" if node.terminal() else "alive",
        int(node.levels_completed),
        None if node.terminal() else avatar_column(frame),
        delta["count"],
        delta["bbox"],
        lattice(frame),
        objects(frame),
    )


def apply(node, route):
    for action in route:
        if node.terminal():
            break
        node.step(*action)


def probe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)

    root_frame = arr(env.frame()).copy()
    print("ROOT", env.actions, color_counts(root_frame), lattice(root_frame), objects(root_frame))
    click = (6, 15, 21)
    routes = {
        "left": [(3,)],
        "right": [(4,)],
        "undo": [(7,)],
        "right_undo": [(4,), (7,)],
        "right_undo_undo": [(4,), (7,), (7,)],
        "click": [click],
        "click_undo": [click, (7,)],
        "click_right": [click, (4,)],
        "click_right_right": [click, (4,), (4,)],
        "right_click": [(4,), click],
        "right_click_right": [(4,), click, (4,)],
        "click_left": [click, (3,)],
        "alt_click": [(7, 15, 21)],
        "click_alt_center": [click, (7, 15, 21)],
        "click_alt_right": [click, (7, 21, 21)],
        "click_click_right": [click, (6, 21, 21)],
    }
    for label, route in routes.items():
        node = env.clone()
        apply(node, route)
        print("CASE", label, route, summary(node, root_frame))

    for action in (6, 7):
        changed = []
        for i, y in enumerate(ROW_ANCHORS):
            for j, x in enumerate(COL_ANCHORS):
                node = env.clone()
                node.step(action, x, y)
                delta = frame_delta(root_frame, node.frame())
                material = sum(
                    1
                    for row in range(63)
                    for col in range(64)
                    if int(root_frame[row][col]) != int(node.frame()[row][col])
                )
                if material:
                    changed.append(((i, j), material, delta["bbox"]))
        print("SCAN", action, changed)


levels, path, err = A.run_program("bp35", probe)
print("RESULT", levels, len(path), err)

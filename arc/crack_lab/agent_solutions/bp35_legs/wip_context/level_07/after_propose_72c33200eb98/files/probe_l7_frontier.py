"""Enumerate compact one-step frontiers from a known safe level-7 route."""

import json
import os
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from legs import BAND, COL_ANCHORS, ROW_ANCHORS, run_actions
from perception import connected_components, frame_delta


L = (3,)
R = (4,)


def click(i, j, kind=6):
    return kind, COL_ANCHORS[j], ROW_ANCHORS[i]


def control(i, kind=6):
    return kind, 3, ROW_ANCHORS[i]


BASE = [
    R, R, R, click(8, 4), control(0), R, control(0), R,
    L, L, L, L, click(9, 2), control(0), (6, 27, 23), control(0), L, L,
    R, click(9, 1), (6, 3, 21), (6, 21, 23), (6, 27, 23), (6, 3, 5),
    R, (6, 33, 39), R, (6, 33, 51), (6, 3, 39), click(4, 4), R,
    (7, 33, 27),
]


def avatar(frame):
    ys, xs = np.where(np.asarray(frame) == 9)
    if not len(xs):
        return None
    return int(round(float(xs.mean()))), int(round(float(ys.mean())))


def cell_shape(frame, i, j):
    color = int(frame[ROW_ANCHORS[i]][COL_ANCHORS[j]])
    r0, c0 = BAND * i, 13 + BAND * j
    area = sum(
        int(frame[row][column]) == color
        for row in range(r0, min(63, r0 + BAND))
        for column in range(c0, c0 + BAND)
    )
    return color, area


def symbol(frame, i, j):
    color, area = cell_shape(frame, i, j)
    if color in (3, 5):
        return "#"
    if color == 10:
        return "."
    if color in (9, 11):
        return "A"
    if color == 12:
        return "X" if area > 5 else "x"
    if color == 15:
        return "P"
    if color == 8:
        return "g"
    if color == 0:
        return "0"
    return "?"


def summary(node):
    frame = node.frame()
    return {
        "terminal": bool(node.terminal()),
        "avatar": avatar(frame),
        "grid": [
            "".join(symbol(frame, row, column) for column in range(8))
            for row in range(10)
        ],
        "supports": [
            (blob.bbox, blob.area)
            for blob in connected_components(frame, colors=(12,), min_area=3)
            if blob.bbox[0] < 63
        ],
        "controls": [
            (blob.bbox, blob.area)
            for blob in connected_components(frame, colors=(8,), min_area=3)
            if blob.bbox[0] < 63
        ],
        "objects": [
            (blob.color, blob.bbox, blob.area)
            for blob in connected_components(
                frame, colors=(7, 8, 9, 11, 12, 14, 15), min_area=3
            )
            if blob.bbox[0] < 63
        ],
    }


def choices(frame):
    position = avatar(frame)
    if position is None:
        return [L, R]
    ax, ay = position
    out = [L, R, (7, 0, 0)]
    controls = [
        blob
        for blob in connected_components(frame, colors=(8,), min_area=3)
        if blob.bbox[1] <= 5 and blob.bbox[0] < 63
    ]
    for blob in controls[:1]:
        y, x = blob.centroid
        out.extend(
            [
                (6, int(round(x)), int(round(y))),
                (7, int(round(x)), int(round(y))),
            ]
        )
    for blob in connected_components(frame, colors=(12,), min_area=3):
        y, x = blob.centroid
        x, y = int(round(x)), int(round(y))
        if abs(x - ax) <= 13 and abs(y - ay) <= 25:
            out.extend([(6, x, y), (7, x, y)])
    for i, y in enumerate(ROW_ANCHORS):
        for j, x in enumerate(COL_ANCHORS):
            if int(frame[y][x]) == 0 and abs(x - ax) <= 13 and abs(y - ay) <= 7:
                out.extend([(6, x, y), (7, x, y)])
    return list(dict.fromkeys(out))


def probe(env):
    with open("checkpoint.json") as stream:
        prefix = json.load(stream)["final_path"]
    for action in prefix:
        env.step(action)
    run_actions(env, BASE)
    stage = os.environ.get("FRONTIER_STAGE", "")
    if stage == "R":
        env.step(*R)
    elif stage == "R_G":
        env.step(*R)
        controls = [
            blob
            for blob in connected_components(env.frame(), colors=(8,), min_area=3)
            if blob.bbox[1] <= 5 and blob.bbox[0] < 63
        ]
        y, x = controls[0].centroid
        env.step(6, int(round(x)), int(round(y)))
    base_level = int(env.levels_completed)
    before = np.asarray(env.frame()).copy()
    print("BASE", {"steps": len(BASE), "stage": stage, **summary(env)})
    for action in choices(before):
        node = env.clone()
        node.step(*action)
        after = np.asarray(node.frame())
        changed = before[:63] != after[:63]
        transitions = {}
        for old, new in zip(before[:63][changed], after[:63][changed]):
            pair = (int(old), int(new))
            transitions[pair] = transitions.get(pair, 0) + 1
        print(
            "TRY",
            action,
            {
                "level_delta": int(node.levels_completed) - base_level,
                "delta": {
                    key: value
                    for key, value in frame_delta(before[:63], node.frame()[:63]).items()
                    if key != "samples"
                },
                "transitions": sorted(transitions.items()),
                **summary(node),
            },
        )
    node = env.clone()
    try:
        node.step(7)
    except Exception as error:
        print("KEY7", {"error": type(error).__name__, "message": str(error)})
    else:
        print("KEY7", summary(node))
    for name, suffix in {
        "7x2": [(7,), (7,)],
        "7x3": [(7,), (7,), (7,)],
        "7_R": [(7,), R],
        "7_R_7": [(7,), R, (7,)],
        "R_7": [R, (7,)],
        "R_7x2": [R, (7,), (7,)],
        "take_left_7": [(6, 21, 29), (7,)],
        "take_below_7": [(6, 33, 35), (7,)],
    }.items():
        node = env.clone()
        run_actions(node, suffix)
        print("SEQ", name, {"suffix": suffix, **summary(node)})


if __name__ == "__main__":
    arena.run_program("bp35", probe)

"""Compact reproduction and one-step probe for the level-7 open-gate state."""
import json
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from legs import MOVES, _avatar_pos, _special_frontier, fast_reach
from perception import connected_components, frame_delta


OPEN = [
    2, 2, 3, 4, 1, 1, 3, 3, 5,
    2, 1, 2, 2, 3, 2, 1, 5,
    2, 5,
    2, 1, 2, 1, 2, 1, 2, 2, 3, 4, 1, 2, 3, 5,
]


def summary(env):
    frame = np.asarray(env.frame())
    helper = connected_components(frame, colors=(14,), min_area=4)
    walls = connected_components(frame, colors=(11, 15), min_area=4)
    reward_path, reach = fast_reach(env)
    return {
        "level": int(env.levels_completed),
        "avatar": _avatar_pos(frame),
        "helper": None if not helper else helper[0].bbox[:2],
        "walls": tuple((b.color, b.bbox, b.area) for b in walls),
        "reach": len(reach),
        "win": reward_path,
        "frontier": tuple(
            (pos, len(path))
            for pos, path in _special_frontier(reach, frame)
        ),
    }


def quick(env):
    frame = np.asarray(env.frame())
    helper = connected_components(frame, colors=(14,), min_area=4)
    marker = [
        b.bbox[:2]
        for b in connected_components(frame, colors=(9,), min_area=4)
        if b.bbox[0] == 1
    ]
    return (
        _avatar_pos(frame),
        None if not helper else helper[0].bbox[:2],
        None if not marker else marker[0],
        int(np.count_nonzero(frame == 15)),
    )


def probe(env):
    with open("checkpoint.json") as fh:
        checkpoint = json.load(fh)["final_path"]
    for action in checkpoint:
        env.step(action)
    root = env.clone()
    print("trace", 0, quick(env))
    for index, action in enumerate(OPEN, 1):
        env.step(action)
        print("trace", index, action, quick(env))
    preuse = root.clone()
    for action in OPEN[:-1]:
        preuse.step(action)
    print("preuse", summary(preuse))
    pre_before = preuse.frame()
    for action in (*MOVES, 5):
        child = preuse.clone()
        child.step(action)
        delta = frame_delta(pre_before, child.frame())
        print(
            "preact", action,
            "delta", (delta["count"], delta["bbox"]),
            "state", summary(child),
        )
    print("open", summary(env))
    before = env.frame()
    for action in (*MOVES, 5):
        child = env.clone()
        child.step(action)
        delta = frame_delta(before, child.frame())
        print(
            "act", action,
            "delta", (delta["count"], delta["bbox"]),
            "state", summary(child),
        )


levels, path, err = arena.run_program("g50t", probe)
print("result", levels, len(path), err)

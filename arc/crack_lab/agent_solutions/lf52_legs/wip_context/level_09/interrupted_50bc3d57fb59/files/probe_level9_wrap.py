"""Trace bounded off-screen carrier motion without frame-state deduplication."""

import json
import sys

import numpy as np

sys.path.insert(0, "/private/tmp/gkm-submitted-protocol-lf52-20260805/arc/crack_lab")

import gkm_arena as arena

from perception import arr, connected_components, safe_step


def carrier(frame):
    return tuple(
        blob.top_left
        for blob in connected_components(frame, colors=(12,))
        if blob.size == (4, 4)
    )


def world_change(before, after):
    return int(np.count_nonzero(arr(before)[1:, :] != arr(after)[1:, :]))


def trace(root, path):
    node = root.clone()
    observations = []
    for index, action in enumerate(path, 1):
        before = arr(node.frame()).copy()
        safe_step(node, action)
        change = world_change(before, node.frame())
        visible = carrier(node.frame())
        if change or visible:
            observations.append((index, action, change, visible))
    return tuple(observations), node


def probe(env):
    with open("checkpoint.json") as handle:
        checkpoint = json.load(handle)
    for action in checkpoint["final_path"]:
        env.step(action)
    root = env.clone()

    for action in (1, 2, 3, 4):
        observations, node = trace(root, (action,) * 24)
        print("RUN", action, observations, "end_level", node.levels_completed)

    for horizontal in (4, 3):
        for count in (3, 4, 6, 8, 10, 12):
            prefix = (horizontal,) * count
            for vertical in (1, 2):
                observations, node = trace(root, prefix + (vertical,) * 12)
                print(
                    "TURN", horizontal, count, vertical,
                    observations, "end_level", node.levels_completed,
                )

    corner_paths = (
        (4, 4, 4, 2),
        (4, 4, 4, 1),
        (4, 4, 4, 2, 3, 3, 3),
        (4, 4, 4, 1, 3, 3, 3),
        (4, 4, 4, 2, 2, 3, 3, 3),
        (4, 4, 4, 1, 1, 3, 3, 3),
    )
    for path in corner_paths:
        observations, node = trace(root, path)
        print("CORNER", path, observations, "end_level", node.levels_completed)

    for vertical in (1, 2):
        for count in range(1, 7):
            path = (4, 4, 4, 4) + (vertical,) * count + (3, 3, 3, 3)
            observations, node = trace(root, path)
            print("OFFSCREEN_TURN", vertical, count, observations, "end_level", node.levels_completed)


levels, path, error = arena.run_program("lf52", probe)
print("PROBE_RESULT", levels, len(path), error)

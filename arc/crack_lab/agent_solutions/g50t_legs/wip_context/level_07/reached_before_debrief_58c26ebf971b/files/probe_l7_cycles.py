"""Test the observed level-7 synchronized two-surface cycle."""
import json
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from legs import _special_frontier, fast_reach
from perception import connected_components


OPEN = [
    2, 2, 3, 4, 1, 1, 3, 3, 5,
    2, 1, 2, 2, 3, 2, 1, 5,
    2, 5,
    2, 1, 2, 1, 2, 1, 2, 2, 3, 4, 1, 2, 3, 5,
]
SYNC = ([2, 1] * 7)[:13]
LEFT = [1, 3, 3, 1, 1, 5]
RIGHT = [2, 2, 4, 4, 1, 1, 1, 1, 3, 1, 1, 4, 4, 4, 5]


def helper(node):
    blobs = connected_components(node.frame(), colors=(14,), min_area=4)
    return None if not blobs else blobs[0].bbox[:2]


def marker(node):
    blobs = connected_components(node.frame(), colors=(9,), min_area=4)
    blob = next((b for b in blobs if b.bbox[0] == 1), None)
    return None if blob is None else blob.bbox[:2]


def report(label, node):
    reward_path, reach = fast_reach(node)
    print(
        label, "level", int(node.levels_completed), "helper", helper(node),
        "marker", marker(node),
        "barrier", int(np.count_nonzero(np.asarray(node.frame()) == 15)),
        "reach", len(reach), "win", reward_path,
        "frontier", tuple(
            (pos, len(path))
            for pos, path in _special_frontier(reach, node.frame())
        ),
        flush=True,
    )
    return reward_path


def probe(env):
    with open("checkpoint.json") as fh:
        checkpoint = json.load(fh)["final_path"]
    for action in checkpoint + OPEN:
        env.step(action)
    for cycle in range(1, 9):
        for action in SYNC:
            env.step(action)
        reward_path = report(("left-ready", cycle), env)
        if reward_path:
            for action in reward_path:
                env.step(action)
            break
        for action in LEFT:
            env.step(action)
        reward_path = report(("left-done", cycle), env)
        if reward_path:
            for action in reward_path:
                env.step(action)
            break
        for action in RIGHT:
            env.step(action)
        reward_path = report(("right-done", cycle), env)
        if reward_path:
            for action in reward_path:
                env.step(action)
            break
        if int(env.levels_completed) > 6 or env.terminal():
            break
    print("end", int(env.levels_completed), helper(env), marker(env))


levels, path, err = arena.run_program("g50t", probe)
print("result", levels, len(path), err)

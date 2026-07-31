"""Scan patrol phases after a concrete level-7 contextual commit."""
import json
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from legs import _special_frontier, fast_reach
from perception import connected_components


ROOT = [
    2, 2, 3, 4, 1, 1, 3, 3, 5,
    2, 1, 2, 2, 3, 2, 1, 5,
    2, 5,
    2, 1, 2, 1, 2, 1, 2, 2, 3, 4, 1, 2, 3, 5,
] + ([2, 1] * 7)[:13] + [1, 3, 3, 1, 1, 5] + [
    2, 2, 4, 4, 1, 1, 1, 1, 3, 1, 1, 4, 4, 4, 5,
]


def helper(node):
    blobs = connected_components(node.frame(), colors=(14,), min_area=4)
    return None if not blobs else blobs[0].bbox[:2]


def probe(env):
    with open("checkpoint.json") as fh:
        checkpoint = json.load(fh)["final_path"]
    for action in checkpoint + ROOT:
        env.step(action)
    node = env.clone()
    for phase in range(33):
        reward_path, reach = fast_reach(node)
        print(
            "phase", phase, "helper", helper(node),
            "barrier", int(np.count_nonzero(np.asarray(node.frame()) == 15)),
            "reach", len(reach), "win", reward_path,
            "frontier", tuple(
                (pos, len(path))
                for pos, path in _special_frontier(reach, node.frame())
            ),
            flush=True,
        )
        if reward_path is not None:
            break
        node.step(2 if phase % 2 == 0 else 1)


levels, path, err = arena.run_program("g50t", probe)
print("result", levels, len(path), err)

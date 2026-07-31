"""Verify the concrete level-7 patrol synchronization and final walk."""
import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

import numpy as np

from legs import _avatar_pos, _special_frontier, fast_reach
from perception import connected_components


OPEN = [
    2, 2, 3, 4, 1, 1, 3, 3, 5,
    2, 1, 2, 2, 3, 2, 1, 5,
    2, 5,
    2, 1, 2, 1, 2, 1, 2, 2, 3, 4, 1, 2, 3, 5,
]
SYNC = ([2, 1] * 7)[:13]
COMMIT = [1, 3, 3, 1, 1, 5]


def helper(env):
    blobs = connected_components(env.frame(), colors=(14,), min_area=4)
    return None if not blobs else blobs[0].bbox[:2]


def summary(env):
    frame = np.asarray(env.frame())
    marker = next(
        (
            b.bbox[:2]
            for b in connected_components(frame, colors=(9,), min_area=4)
            if b.bbox[0] == 1
        ),
        None,
    )
    reward_path, reach = fast_reach(env)
    return (
        int(env.levels_completed), _avatar_pos(frame), helper(env), marker,
        int(np.count_nonzero(frame == 15)), len(reach), reward_path,
        tuple((pos, len(path)) for pos, path in _special_frontier(
            reach, frame
        )),
    )


def probe(env):
    with open("checkpoint.json") as fh:
        checkpoint = json.load(fh)["final_path"]
    for action in checkpoint + OPEN + SYNC:
        env.step(action)
    print("synced", summary(env))
    for action in COMMIT:
        env.step(action)
    print("committed", summary(env))
    for action in (1, 2, 3, 4, 5):
        child = env.clone()
        child.step(action)
        print("act", action, summary(child))


levels, path, err = arena.run_program("g50t", probe)
print("result", levels, len(path), err)

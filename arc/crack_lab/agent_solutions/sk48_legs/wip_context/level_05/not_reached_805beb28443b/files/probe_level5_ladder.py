"""Trace minimal tether-ladder candidates around the level-5 block."""
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena
import numpy as np

import perception as p
import players


def state(env):
    data = p.arr(env.frame())
    avatar = np.argwhere(data[:53, :11] == 6)
    top = int(avatar[:, 0].min())
    pieces = tuple(
        (blob.color, blob.bbox[0], blob.bbox[1])
        for blob in p.connected_components(
            env.frame(), colors=(8, 9), min_area=4
        )
        if blob.bbox[0] < 53
    )
    tether = tuple(
        blob.bbox
        for blob in p.connected_components(
            env.frame(), colors=(2,), min_area=2
        )
        if blob.bbox[0] < 53 and blob.bbox[1] > 9
    )
    return top, pieces, tether


def run(env):
    for level in range(1, 5):
        getattr(players, f"play_level_{level}")(env)
    for extension in range(0, 6):
        path = [1] + [4] * extension + [3, 2]
        branch = p.replay(env, path)
        print("LADDER", extension, path, state(branch))
        for suffix in (
            [4],
            [4, 4],
            [3],
            [1],
            [2],
            [4, 1],
            [4, 2],
            [3, 1],
            [3, 2],
        ):
            after = p.replay(branch, suffix)
            if state(after) != state(branch):
                print("NEXT", extension, suffix, state(after))


levels, path, error = arena.run_program("sk48", run)
print("END", levels, len(path), error)

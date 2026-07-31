import json
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from legs import _avatar_pos
from perception import connected_components


PREFIX = (
    [2, 1, 2, 1, 2, 2, 3, 5, 2, 1, 2, 1, 2]
    + [1, 3, 3, 1, 1, 5]
    + [2, 1, 2, 1, 2, 2, 3, 5]
    + [2, 1, 2, 2, 3, 5] + [2, 1] * 5
    + [2, 1] * 5 + [3, 3, 1, 1, 5] + [2]
)
SUFFIX = [2, 3, 5, 2, 2, 3, 5, 2, 1, 2, 1]


def state(env):
    f = np.asarray(env.frame())
    avatar = _avatar_pos(f)
    helper = next(
        (b.bbox[:2] for b in connected_components(
            f, colors=(14,), min_area=4)),
        None,
    )
    specials = tuple(
        (b.color, b.bbox, b.area)
        for b in connected_components(f, colors=(11, 15), min_area=4)
    )
    return (
        int(env.levels_completed),
        bool(env.terminal()),
        avatar,
        helper,
        specials,
    )


def probe(env):
    with open("checkpoint.json") as fh:
        checkpoint = json.load(fh)["final_path"]
    for action in checkpoint + PREFIX:
        env.step(action)
    print("start", state(env))
    last = state(env)
    for index, action in enumerate(SUFFIX, 1):
        if env.terminal():
            break
        env.step(action)
        now = state(env)
        print(index, action, now)
        last = now


levels, path, err = arena.run_program("g50t", probe)
print("result", levels, len(path), err)

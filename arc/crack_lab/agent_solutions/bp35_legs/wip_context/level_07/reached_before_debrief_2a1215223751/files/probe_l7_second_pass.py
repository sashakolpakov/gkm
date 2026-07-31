"""Verify whether the staged first pass makes the right-shaft descent safe."""

import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from probe_l7_raw_search import SEED, avatar_position
from perception import connected_components


ROUTE = [
    (6, 39, 51),
    (6, 3, 3),
    (4,),
    (7, 0, 0),
    (4,),
    (6, 3, 3),
    (3,),
    (3,),
    (3,),
    (3,),
    (3,),
]


def actors(frame):
    return [
        (blob.color, blob.bbox, blob.area)
        for blob in connected_components(
            frame, colors=(7, 9, 12, 14, 15), min_area=3
        )
        if blob.bbox[0] < 63
    ]


def probe(env):
    with open("checkpoint.json") as stream:
        prefix = json.load(stream)["final_path"]
    for action in prefix:
        env.step(action)
    base_level = int(env.levels_completed)
    for action in SEED:
        env.step(*action)
    print("staged", len(SEED), avatar_position(env.frame()), actors(env.frame()))
    for index, action in enumerate(ROUTE, 1):
        env.step(*action)
        print(
            index,
            action,
            {
                "level_delta": int(env.levels_completed) - base_level,
                "terminal": bool(env.terminal()),
                "avatar": avatar_position(env.frame()),
                "actors": actors(env.frame()),
            },
        )
        if env.terminal() or env.levels_completed > base_level:
            break


arena.run_program("bp35", probe)

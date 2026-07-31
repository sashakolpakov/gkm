"""Test the reproduced four-action staircase macro to reward or failure."""

import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from probe_l7_raw_search import SEED, avatar_position
from perception import connected_components


MACRO = [(7, 39, 38), (7, 39, 32), (6, 3, 47), (3,)]


def probe(env):
    with open("checkpoint.json") as stream:
        prefix = json.load(stream)["final_path"]
    for action in prefix:
        env.step(action)
    base_level = int(env.levels_completed)
    route = list(SEED)
    for action in route:
        env.step(*action)
        if env.terminal() or env.levels_completed > base_level:
            break
    print(
        "seed",
        {
            "len": len(route),
            "level_delta": int(env.levels_completed) - base_level,
            "terminal": bool(env.terminal()),
            "avatar": avatar_position(env.frame()),
            "actors": [
                (blob.color, blob.bbox, blob.area)
                for blob in connected_components(
                    env.frame(), colors=(7, 9, 12, 14, 15), min_area=3
                )
                if blob.bbox[0] < 63
            ],
            "lattice": [
                [int(env.frame()[3 + 6 * i][15 + 6 * j]) for j in range(8)]
                for i in range(10)
            ],
        },
    )
    for repetition in range(1, int(os.environ.get("REPEATS", "30")) + 1):
        if env.terminal() or env.levels_completed > base_level:
            break
        for action in MACRO:
            env.step(*action)
            route.append(action)
            if env.terminal() or env.levels_completed > base_level:
                break
        print(
            "repeat",
            repetition,
            {
                "len": len(route),
                "level_delta": int(env.levels_completed) - base_level,
                "terminal": bool(env.terminal()),
                "avatar": avatar_position(env.frame()),
            },
        )
    print("route", route)


arena.run_program("bp35", probe)

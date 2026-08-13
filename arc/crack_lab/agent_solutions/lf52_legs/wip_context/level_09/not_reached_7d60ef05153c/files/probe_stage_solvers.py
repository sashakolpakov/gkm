"""Run the exact visible movable-bridge solver at admitted stage boundaries."""

import json
import os

import gkm_try
from legs import _movable_bridge_board, _movable_bridge_solution


TARGET_LEVEL = int(os.environ.get("TARGET_LEVEL", "6"))
TARGET_GROUP = int(os.environ.get("TARGET_GROUP", "11"))
LEVEL_ENDS = {1: 8, 2: 42, 3: 87, 4: 149, 5: 238, 6: 331, 7: 476, 8: 544}


def probe(env):
    with open("checkpoint.json") as checkpoint_file:
        path = json.load(checkpoint_file)["final_path"]
    start = LEVEL_ENDS[TARGET_LEVEL - 1]
    end = LEVEL_ENDS[TARGET_LEVEL]
    for action in path[:start]: env.step(action)
    segment = path[start:end]
    if TARGET_GROUP < 0:
        index = group = 0
        print("STAGE_TRACE", group, _movable_bridge_board(env.frame()))
        while index < len(segment):
            while index < len(segment) and not isinstance(segment[index], list):
                env.step(segment[index]); index += 1
            count = 0
            while index < len(segment) and isinstance(segment[index], list) and count < 2:
                env.step(*segment[index]); index += 1; count += 1
            group += 1
            print("STAGE_TRACE", group, _movable_bridge_board(env.frame()))
        return
    index = group = 0
    while index < len(segment) and group <= TARGET_GROUP:
        while index < len(segment) and not isinstance(segment[index], list):
            env.step(segment[index]); index += 1
        if group == TARGET_GROUP:
            break
        count = 0
        while index < len(segment) and isinstance(segment[index], list) and count < 2:
            env.step(*segment[index]); index += 1; count += 1
        group += 1
    print(
        "STAGE_SOLVER", TARGET_LEVEL, TARGET_GROUP,
        _movable_bridge_board(env.frame()), _movable_bridge_solution(env.frame()),
    )


gkm_try.A.run_program("lf52", probe)

"""Compact symbolic trace of one admitted checkpoint level."""

import json
import os

import gkm_try
from legs import _movable_bridge_board
from perception import arr, connected_components


TARGET_LEVEL = int(os.environ.get("TARGET_LEVEL", "7"))
LEVEL_ENDS = {1: 8, 2: 42, 3: 87, 4: 149, 5: 238, 6: 331, 7: 476, 8: 544}


def is_click(action):
    return isinstance(action, (list, tuple)) and len(action) == 3


def compact_state(node):
    slots, carriers, bridges, pegs = _movable_bridge_board(node.frame())
    extra = {}
    for blob in connected_components(node.frame(), colors=(3, 9, 10, 11, 12, 14, 15)):
        if blob.area >= 4:
            extra.setdefault(blob.color, []).append((blob.top_left, blob.size, blob.area))
    return {
        "C": tuple(sorted(carriers)),
        "B": tuple(sorted(bridges)),
        "P": tuple(sorted(pegs)),
        "X": tuple((color, tuple(values)) for color, values in sorted(extra.items())),
        "h": hash(arr(node.frame())[1:, :].tobytes()),
    }


def trace(env):
    with open("checkpoint.json") as checkpoint_file:
        path = json.load(checkpoint_file)["final_path"]
    start = LEVEL_ENDS[TARGET_LEVEL - 1]
    end = LEVEL_ENDS[TARGET_LEVEL]
    for action in path[:start]:
        env.step(action)
    segment = path[start:end]
    print("TRACE_ENTRY", TARGET_LEVEL, compact_state(env))
    index = 0
    group = 0
    while index < len(segment):
        keys = []
        while index < len(segment) and not is_click(segment[index]):
            keys.append(segment[index])
            env.step(segment[index])
            index += 1
        print("TRACE_KEYS", group, "".join(map(str, keys)), compact_state(env))
        clicks = []
        while index < len(segment) and is_click(segment[index]) and len(clicks) < 2:
            clicks.append(tuple(segment[index]))
            env.step(segment[index])
            index += 1
        if clicks:
            print("TRACE_MOVE", group, clicks, compact_state(env), env.levels_completed)
        group += 1


gkm_try.A.run_program("lf52", trace)

"""Compact bridge/carrier state trace for checkpoint levels four and five."""

import json
import os

import gkm_try
from legs import _bridge_carrier_moves, _bridge_carrier_state


TARGET_LEVEL = int(os.environ.get("TARGET_LEVEL", "5"))
LEVEL_ENDS = {1: 8, 2: 42, 3: 87, 4: 149, 5: 238, 6: 331, 7: 476, 8: 544}


def is_click(action):
    return isinstance(action, (list, tuple)) and len(action) == 3


def state(frame):
    slots, pegs, carriers, bridges, borders, selected = _bridge_carrier_state(frame)
    return (
        "P", tuple(sorted(pegs)), "C", tuple(sorted(carriers)),
        "B", tuple(sorted(bridges)), "S", len(slots), "D", tuple(sorted(borders)),
    )


def trace(env):
    with open("checkpoint.json") as checkpoint_file:
        path = json.load(checkpoint_file)["final_path"]
    start = LEVEL_ENDS[TARGET_LEVEL - 1]
    end = LEVEL_ENDS[TARGET_LEVEL]
    for action in path[:start]:
        env.step(action)
    segment = path[start:end]
    print("BRIDGE_ENTRY", TARGET_LEVEL, state(env.frame()))
    index = group = 0
    while index < len(segment):
        keys = []
        while index < len(segment) and not is_click(segment[index]):
            keys.append(segment[index]); env.step(segment[index]); index += 1
        print(
            "BRIDGE_KEYS", group, "".join(map(str, keys)),
            "LEGAL", _bridge_carrier_moves(env.frame()), state(env.frame()),
        )
        clicks = []
        while index < len(segment) and is_click(segment[index]) and len(clicks) < 2:
            clicks.append(tuple(segment[index])); env.step(segment[index]); index += 1
        if clicks:
            print("BRIDGE_MOVE", group, clicks, state(env.frame()), env.levels_completed)
        group += 1


gkm_try.A.run_program("lf52", trace)

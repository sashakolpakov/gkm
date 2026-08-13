"""Compact symbolic trace of one validated checkpoint level."""

import json
import os

import gkm_try
from legs import _bridge_carrier_state


TARGET_LEVEL = int(os.environ.get("TARGET_LEVEL", "5"))
LEVEL_ENDS = {1: 8, 2: 42, 3: 87, 4: 149, 5: 238, 6: 331, 7: 476, 8: 544}


def is_coordinate(action):
    return isinstance(action, (list, tuple)) and len(action) == 3 and action[0] == 6


def compact(frame):
    slots, pegs, carriers, bridges, borders, selected = _bridge_carrier_state(frame)
    return {
        "pegs": tuple(sorted(pegs)),
        "carriers": tuple(sorted(carriers)),
        "bridges": tuple(sorted(bridges)),
        "borders": tuple(sorted(borders)),
        "slots": len(slots),
        "selected": selected,
    }


def trace(env):
    with open("checkpoint.json") as checkpoint_file:
        full_path = json.load(checkpoint_file)["final_path"]
    start = LEVEL_ENDS[TARGET_LEVEL - 1]
    end = LEVEL_ENDS[TARGET_LEVEL]
    for action in full_path[:start]:
        env.step(action)
    print("TRACE_ENTRY", TARGET_LEVEL, compact(env.frame()))
    actions = full_path[start:end]
    index = 0
    macros = []
    while index < len(actions):
        keys = []
        while index < len(actions) and not is_coordinate(actions[index]):
            keys.append(actions[index])
            env.step(actions[index])
            index += 1
        before = compact(env.frame())
        clicks = []
        while index < len(actions) and is_coordinate(actions[index]) and len(clicks) < 2:
            clicks.append(actions[index])
            env.step(actions[index])
            index += 1
        after = compact(env.frame())
        move = tuple((click[2] - 1, click[1] - 1) for click in clicks)
        macros.append((tuple(keys), move, before, after, env.levels_completed))
    print("TRACE_MACROS", macros)


gkm_try.A.run_program("lf52", trace)

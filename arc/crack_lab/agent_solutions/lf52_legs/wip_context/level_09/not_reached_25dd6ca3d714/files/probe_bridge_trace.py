"""Compact symbolic trace of admitted bridge-carrier level paths."""

import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from legs import _bridge_carrier_state


BOUNDARIES = (0, 8, 42, 87, 149, 238, 331, 476, 544)


def observe(env):
    level = int(os.environ.get("TARGET_LEVEL", "5"))
    with open("checkpoint.json") as stream:
        path = json.load(stream)["final_path"]
    start, end = BOUNDARIES[level - 1], BOUNDARIES[level]
    for action in path[:start]:
        env.step(action)
    root_state = _bridge_carrier_state(env.frame())
    print("ROOT", {"pegs": tuple(sorted(root_state[1])),
                   "carriers": tuple(sorted(root_state[2])),
                   "bridges": tuple(sorted(root_state[3])),
                   "slots": tuple(sorted(root_state[0])),
                   "borders": tuple(sorted(root_state[4]))})
    token_index = 0
    action_index = 0
    while action_index < end - start:
        action = path[start + action_index]
        if isinstance(action, list):
            token = path[start + action_index:start + action_index + 2]
            action_index += 2
        else:
            token = [action]
            action_index += 1
        for item in token:
            env.step(item)
        token_index += 1
        if len(token) == 2 or env.levels_completed >= level:
            state = _bridge_carrier_state(env.frame())
            print("TOKEN", {"token": token_index, "actions": action_index,
                            "kind": "macro" if len(token) == 2 else "key",
                            "pegs": tuple(sorted(state[1])),
                            "carriers": tuple(sorted(state[2])),
                            "bridges": tuple(sorted(state[3])),
                            "level": env.levels_completed})


levels, path, error = arena.run_program("lf52", observe)
print("PROBE_RESULT", {"levels": levels, "moves": len(path), "error": str(error)})

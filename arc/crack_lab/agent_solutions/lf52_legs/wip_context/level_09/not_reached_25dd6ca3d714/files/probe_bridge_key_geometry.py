"""Compact carrier/fixed-bridge states after each admitted level-5 key."""

import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from legs import _bridge_carrier_state
from perception import safe_step


def compact(node):
    state = _bridge_carrier_state(node.frame())
    return {"pegs": tuple(sorted(state[1])),
            "carriers": tuple(sorted(state[2])),
            "fixed": tuple(sorted(state[3])),
            "borders": tuple(sorted(state[4]))}


def observe(env):
    with open("checkpoint.json") as stream:
        path = json.load(stream)["final_path"]
    for action in path[:149]:
        safe_step(env, action)
    print("ENTRY", compact(env))
    macro = 0
    click_count = 0
    key_in_group = 0
    for action in path[149:238]:
        safe_step(env, action)
        if isinstance(action, list):
            click_count += 1
            if click_count == 2:
                macro += 1
                click_count = 0
                key_in_group = 0
                print("MACRO", macro, compact(env))
        else:
            key_in_group += 1
            print("KEY", {"before_macro": macro + 1,
                          "index": key_in_group, "action": action,
                          **compact(env)})


levels, path, error = arena.run_program("lf52", observe)
print("PROBE_RESULT", {"levels": levels, "moves": len(path),
                       "error": str(error)})

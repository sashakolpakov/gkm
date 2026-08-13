"""Check whether level-7 carrier and bridge tiles alter key control."""

import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import arr, safe_step


def key(node):
    return arr(node.frame())[1:, :].tobytes()


def observe(env):
    with open("checkpoint.json") as stream:
        path = json.load(stream)["final_path"]
    for action in path[:331]:
        safe_step(env, action)

    root_key = key(env)
    contexts = {
        "carrier": (6, 37, 37),
        "fixed_left": (6, 7, 19),
        "fixed_right": (6, 43, 19),
        "movable_bridge": (6, 43, 13),
        "peg": (6, 7, 13),
        "empty_slot": (6, 13, 55),
    }
    for name, click in contexts.items():
        clicked = env.clone()
        safe_step(clicked, click)
        click_changed = key(clicked) != root_key
        for action in (1, 2, 3, 4):
            branch = clicked.clone()
            safe_step(branch, action)
            direct = env.clone()
            safe_step(direct, action)
            print("INTERACTION", {"name": name, "key": action,
                                  "click_changed": click_changed,
                                  "same_as_direct": key(branch) == key(direct)})


levels, path, error = arena.run_program("lf52", observe)
print("PROBE_RESULT", {"levels": levels, "moves": len(path),
                       "error": str(error)})

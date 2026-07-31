import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A

import legs


def probe(env):
    with open("checkpoint.json") as source:
        prefix = json.load(source)["final_path"]
    for action in prefix:
        env.step(*action) if isinstance(action, list) else env.step(action)
    path = legs.gravity_room_search(env, max_states=500, debug=True)
    print("MACRO_RESULT", len(path), path)


if __name__ == "__main__":
    A.run_program("bp35", probe)

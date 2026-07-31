import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A

from legs import _bridge_carrier_moves, _bridge_carrier_state


PREFIX = (
    (6, 13, 25), (6, 25, 25),
    3, 3, 2, 2, 2,
    (6, 25, 25), (6, 37, 25),
    4, 4, 4, 2, 2, 2,
    (6, 43, 25), (6, 31, 25),
)
NAVIGATION = (
    2, 2, 3, 2, 4, 4, 4, 1, 1, 1, 1, 4, 4, 4, 2,
)
CAPTURE = ((6, 31, 25), (6, 43, 25))


def act(env, action):
    env.step(*action) if isinstance(action, tuple) else env.step(action)


def probe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    assert checkpoint["game"] == "lf52" and checkpoint["validated"]
    for action in checkpoint["final_path"]:
        act(env, tuple(action) if isinstance(action, list) else action)
    for action in PREFIX + NAVIGATION:
        act(env, action)
    print("ALIGNED", env.levels_completed, _bridge_carrier_state(env.frame()))
    print("MOVES", _bridge_carrier_moves(env.frame()))
    for action in CAPTURE:
        act(env, action)
    print("AFTER", env.levels_completed, _bridge_carrier_state(env.frame()))
    print("AFTER_MOVES", _bridge_carrier_moves(env.frame()))


levels, path, err = A.run_program("lf52", probe)
print("END", levels, len(path), err)

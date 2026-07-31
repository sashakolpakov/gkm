import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A

from legs import _bridge_carrier_moves, _bridge_carrier_state


FIRST_CAPTURE = (
    (6, 13, 25), (6, 25, 25),
    3, 3, 2, 2, 2,
    (6, 25, 25), (6, 37, 25),
    4, 4, 4, 2, 2, 2,
    (6, 43, 25), (6, 31, 25),
)
NAVIGATION = (2, 2, 3, 2, 4, 4, 4, 1, 1, 1, 1, 4, 4, 4, 2)


def act(env, action):
    env.step(*action) if isinstance(action, tuple) else env.step(action)


def compact(env):
    slots, pegs, carriers, bridges, borders, _ = _bridge_carrier_state(env.frame())
    return (
        len(slots), tuple(sorted(pegs)), tuple(sorted(carriers)),
        tuple(sorted(bridges)), tuple(sorted(borders)),
        _bridge_carrier_moves(env.frame()),
    )


def probe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        act(env, tuple(action) if isinstance(action, list) else action)
    for action in FIRST_CAPTURE:
        act(env, action)
    print("STEP", 0, compact(env))
    for index, action in enumerate(NAVIGATION, 1):
        env.step(action)
        print("STEP", index, "action", action, compact(env))


levels, path, err = A.run_program("lf52", probe)
print("END", levels, len(path), err)

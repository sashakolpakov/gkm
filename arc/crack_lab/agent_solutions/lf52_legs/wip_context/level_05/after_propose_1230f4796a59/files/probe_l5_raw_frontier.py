import json
import sys
from collections import deque

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A

from legs import _bridge_carrier_moves, _bridge_carrier_state
from perception import arr


PREFIX = (
    (6, 13, 25), (6, 25, 25),
    3, 3, 2, 2, 2,
    (6, 25, 25), (6, 37, 25),
    4, 4, 4, 2, 2, 2,
    (6, 43, 25), (6, 31, 25),
)


def act(env, action):
    env.step(*action) if isinstance(action, tuple) else env.step(action)


def key(env):
    return arr(env.frame())[1:, :].tobytes()


def compact(env):
    slots, pegs, carriers, bridges, borders, selected = _bridge_carrier_state(
        env.frame()
    )
    return (
        tuple(sorted(pegs)), tuple(sorted(carriers)),
        tuple(sorted(bridges)), tuple(sorted(borders)), len(slots),
    )


def probe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    assert checkpoint["game"] == "lf52" and checkpoint["validated"]
    for action in checkpoint["final_path"]:
        act(env, tuple(action) if isinstance(action, list) else action)
    for action in PREFIX:
        act(env, action)
    queue = deque([(env.clone(), ())])
    seen = {key(env)}
    frontiers = []
    while queue and len(seen) < 800:
        node, path = queue.popleft()
        moves = _bridge_carrier_moves(node.frame())
        if moves:
            frontiers.append((path, compact(node), moves))
        for action in (1, 2, 3, 4):
            child = node.clone()
            child.step(action)
            child_key = key(child)
            if child_key in seen:
                continue
            seen.add(child_key)
            queue.append((child, path + (action,)))
    print("RAW", len(seen), "QUEUE", len(queue), "MOVE_STATES", len(frontiers))
    for item in frontiers[:30]:
        print("FRONTIER", item)


levels, path, err = A.run_program("lf52", probe)
print("END", levels, len(path), err)

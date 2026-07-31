import json
import sys
from collections import deque

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A

from legs import _bridge_carrier_moves, _bridge_carrier_state


TO_LARGE_BOARD = (
    (6, 13, 25), (6, 25, 25),
    3, 3, 2, 2, 2,
    (6, 25, 25), (6, 37, 25),
    4, 4, 4, 2, 2, 2,
    (6, 43, 25), (6, 31, 25),
    2, 2, 3, 2, 4, 4, 4, 1, 1, 1, 1, 4, 4, 4, 2,
    (6, 31, 25), (6, 43, 25),
)

STAGE = TO_LARGE_BOARD + (
    3, 3, 3, 3, 3, 3, 3, 1, 1,
    (6, 10, 25), (6, 22, 25),
    (6, 22, 25), (6, 34, 25),
    2, 2, 4, 4, 4, 4, 1, 1,
    (6, 34, 25), (6, 46, 25),
    (6, 46, 31), (6, 46, 19),
    1, 1, 4,
    (6, 46, 19), (6, 46, 7),
    (6, 52, 7), (6, 40, 7),
    (6, 40, 7), (6, 28, 7),
    3, 3, 3,
    (6, 28, 7), (6, 28, 19),
    (6, 28, 19), (6, 28, 31),
    4, 4, 2, 2, 2, 2, 3, 3,
    (6, 28, 31), (6, 28, 43),
    (6, 28, 43), (6, 28, 55),
    (6, 34, 55), (6, 22, 55),
)


def act(env, action):
    env.step(*action) if isinstance(action, tuple) else env.step(action)


def replay(root, path):
    node = root.clone()
    for action in path:
        node.step(action)
    return node


def probe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    assert checkpoint["game"] == "lf52" and checkpoint["validated"]
    for action in checkpoint["final_path"]:
        act(env, tuple(action) if isinstance(action, list) else action)
    for action in STAGE:
        act(env, action)

    root = env.clone()
    queue = deque([()])
    seen = {_bridge_carrier_state(root.frame())}
    frontiers = {}
    max_depth = 36
    while queue and len(seen) < 1000:
        path = queue.popleft()
        node = replay(root, path)
        moves = _bridge_carrier_moves(node.frame())
        for move in moves:
            frontiers.setdefault(move, (path, len(seen)))
        if len(path) >= max_depth:
            continue
        for action in (4, 3, 2, 1):
            child = node.clone()
            child.step(action)
            state = _bridge_carrier_state(child.frame())
            if state in seen:
                continue
            seen.add(state)
            queue.append(path + (action,))

    state = _bridge_carrier_state(root.frame())
    print("ROOT", "pegs", tuple(sorted(state[1])), "bridges", tuple(sorted(state[3])))
    print("SEARCH", len(seen), "queue", len(queue), "frontiers", len(frontiers))
    for move, (path, discovered) in sorted(frontiers.items()):
        print("FRONTIER", move, "depth", len(path), "path", path, "seen", discovered)


levels, path, err = A.run_program("lf52", probe)
print("END", levels, len(path), err)

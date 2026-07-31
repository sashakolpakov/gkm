import heapq
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


def act(env, action):
    env.step(*action) if isinstance(action, tuple) else env.step(action)


def priority(state, path):
    slots, pegs = state[:2]
    distances = [
        abs(a[0] - b[0]) + abs(a[1] - b[1])
        for index, a in enumerate(pegs)
        for b in tuple(pegs)[index + 1:]
    ]
    pair_distance = min(distances) if distances else 1000
    return (-len(pegs), pair_distance, -len(slots), len(path))


def probe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        act(env, tuple(action) if isinstance(action, list) else action)
    for action in FIRST_CAPTURE:
        act(env, action)

    root = env.clone()
    root_state = _bridge_carrier_state(root.frame())
    serial = 0
    queue = [(priority(root_state, ()), serial, ())]
    seen = {root_state}
    while queue and len(seen) < 650:
        _, _, path = heapq.heappop(queue)
        for action in (4, 3, 2, 1):
            child_path = path + (action,)
            child = root.clone()
            for replay_action in child_path:
                child.step(replay_action)
            state = _bridge_carrier_state(child.frame())
            if state in seen:
                continue
            seen.add(state)
            moves = _bridge_carrier_moves(child.frame())
            if moves:
                print("FOUND", len(seen), len(child_path), child_path, moves)
                return
            if len(child_path) < 16:
                serial += 1
                heapq.heappush(
                    queue, (priority(state, child_path), serial, child_path)
                )
    print("STOP", len(seen), len(queue))


levels, path, err = A.run_program("lf52", probe)
print("END", levels, len(path), err)

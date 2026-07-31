import json
import sys
from collections import deque

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A

from legs import _bridge_carrier_moves, _bridge_carrier_state
from perception import frame_delta


PREFIX = (
    (6, 13, 25), (6, 25, 25),
    3, 3, 2, 2, 2,
    (6, 25, 25), (6, 37, 25),
    4, 4, 4, 2, 2, 2,
    (6, 43, 25), (6, 31, 25),
)


def act(env, action):
    env.step(*action) if isinstance(action, tuple) else env.step(action)


def compact(env):
    slots, pegs, carriers, bridges, borders, selected = _bridge_carrier_state(
        env.frame()
    )
    return (
        tuple(sorted(pegs)), tuple(sorted(carriers)), tuple(sorted(bridges)),
        tuple(sorted(borders)), selected, len(slots),
    )


def probe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    assert checkpoint["game"] == "lf52" and checkpoint["validated"]
    for action in checkpoint["final_path"]:
        act(env, tuple(action) if isinstance(action, list) else action)
    for action in PREFIX:
        act(env, action)
    base_level = env.levels_completed
    print("ROOT", compact(env))
    tests = (
        (1,), (2,), (3,), (4,),
        (1, 1, 1), (2, 2, 2), (3, 3, 3), (4, 4, 4),
        (1, 1, 1, 3, 3, 3), (1, 1, 1, 4, 4, 4),
        (2, 2, 2, 3, 3, 3), (2, 2, 2, 4, 4, 4),
        (3, 3, 3, 1, 1, 1), (4, 4, 4, 1, 1, 1),
        (3, 3, 3, 2, 2, 2), (4, 4, 4, 2, 2, 2),
        (1, 1, 1, 4),
        (1, 1, 1, 4, 2),
        (1, 1, 1, 4, 2, 2),
        (1, 1, 1, 4, 2, 2, 2),
        (1, 1, 1, 3, 2, 2, 2),
    )
    for path in tests:
        child = env.clone()
        for action in path:
            child.step(action)
        print("TEST", path, compact(child), _bridge_carrier_moves(child.frame()))
    for path in ((), (1,), (1, 1), (1, 1, 1), (2,), (2, 2), (2, 2, 2)):
        child = env.clone()
        for action in path:
            child.step(action)
        peg = compact(child)[0][0]
        before = child.frame()
        child.step(6, peg[1] + 1, peg[0] + 1)
        print(
            "SELECT", path, peg,
            frame_delta(before, child.frame())["count"],
            compact(child),
        )

    queue = deque([(env.clone(), ())])
    seen = {compact(env)}
    rewards = []
    move_states = []
    while queue and len(seen) < 240:
        node, path = queue.popleft()
        moves = _bridge_carrier_moves(node.frame())
        if moves:
            move_states.append((path, compact(node), moves))
        if len(path) >= 12:
            continue
        for action in (1, 2, 3, 4):
            child = node.clone()
            child.step(action)
            child_path = path + (action,)
            if child.levels_completed > base_level:
                rewards.append(child_path)
                queue.clear()
                break
            key = compact(child)
            if key not in seen:
                seen.add(key)
                queue.append((child, child_path))
    print("BFS", len(seen), "REWARD", rewards[:1])
    print("MOVE_STATES", move_states[:12])


levels, path, err = A.run_program("lf52", probe)
print("END", levels, len(path), err)

import json
import sys
from collections import deque

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A

from legs import _bridge_carrier_moves, _bridge_carrier_state


def act(env, action):
    env.step(*action) if isinstance(action, tuple) else env.step(action)


def choose(frame, blocked):
    moves = _bridge_carrier_moves(frame)
    captures = [move for move in moves if move[0] == "capture"]
    if captures:
        return captures[0]
    bridges = [
        move for move in moves
        if (move[1], move[2]) not in blocked
    ]
    return bridges[0] if bridges else None


def alignment(root, blocked, limit=900):
    queue = deque([(root.clone(), ())])
    seen = {_bridge_carrier_state(root.frame())}
    while queue and len(seen) < limit:
        node, path = queue.popleft()
        for action in (4, 3, 2, 1):
            child = node.clone()
            child.step(action)
            child_path = path + (action,)
            state = _bridge_carrier_state(child.frame())
            if state in seen:
                continue
            seen.add(state)
            move = choose(child.frame(), blocked)
            if move is not None:
                return child_path, move, len(seen)
            if len(child_path) < 28:
                queue.append((child, child_path))
    return None, None, len(seen)


def probe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    assert checkpoint["game"] == "lf52" and checkpoint["validated"]
    for action in checkpoint["final_path"]:
        act(env, tuple(action) if isinstance(action, list) else action)
    base = env.levels_completed
    blocked = set()
    for index in range(50):
        if env.levels_completed > base:
            break
        move = choose(env.frame(), blocked)
        if move is None:
            path, move, states = alignment(env, blocked)
            print("ALIGN", index, states, path, move)
            if path is None:
                break
            for action in path:
                env.step(action)
        kind, source, destination = move
        pegs_before = len(_bridge_carrier_state(env.frame())[1])
        env.step(6, source[1] + 1, source[0] + 1)
        env.step(6, destination[1] + 1, destination[0] + 1)
        pegs_after = len(_bridge_carrier_state(env.frame())[1])
        print(
            "MACRO", index, kind, source, destination,
            "PEGS", pegs_before, pegs_after,
            "LEVEL", env.levels_completed,
        )
        if kind == "bridge":
            blocked.add((destination, source))
        else:
            blocked.clear()
    print("FINAL", env.levels_completed, _bridge_carrier_state(env.frame()))


levels, path, err = A.run_program("lf52", probe)
print("END", levels, len(path), err)

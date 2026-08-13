"""Enumerate key-reachable legal moves at admitted bridge-carrier junctions."""

import json
import os
import sys
from collections import deque

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from legs import _bridge_carrier_moves
from perception import arr, safe_step


BOUNDARIES = (0, 8, 42, 87, 149, 238)
KEYS = (1, 2, 3, 4)


def state_key(node):
    return int(node.levels_completed), arr(node.frame())[1:, :].tobytes()


def observe(env):
    level = int(os.environ.get("TARGET_LEVEL", "5"))
    after_macros = int(os.environ.get("AFTER_MACROS", "0"))
    with open("checkpoint.json") as stream:
        admitted = json.load(stream)["final_path"]
    start = BOUNDARIES[level - 1]
    for action in admitted[:start]:
        safe_step(env, action)
    coordinate_actions = 0
    for action in admitted[start:BOUNDARIES[level]]:
        if coordinate_actions >= 2 * after_macros:
            break
        safe_step(env, action)
        if isinstance(action, list):
            coordinate_actions += 1
    if os.environ.get("HORIZONTAL_BRANCH") == "1":
        for action in (1, (6, 46, 19), (6, 34, 19)):
            safe_step(env, action)

    root = env.clone()
    max_depth = int(os.environ.get("MAX_DEPTH", "16"))
    max_states = int(os.environ.get("MAX_STATES", "700"))
    queue = deque([(root, ())])
    seen = {state_key(root)}
    options = {}
    while queue and len(seen) <= max_states:
        node, path = queue.popleft()
        for move in _bridge_carrier_moves(node.frame()):
            options.setdefault(move, path)
        if len(path) >= max_depth:
            continue
        for action in KEYS:
            child = node.clone()
            safe_step(child, action)
            key = state_key(child)
            if key not in seen:
                seen.add(key)
                queue.append((child, path + (action,)))

    print("BRIDGE_OPTIONS", {"level": level, "after_macros": after_macros,
                             "states": len(seen), "remaining": len(queue)})
    for move, path in sorted(options.items(), key=lambda item:
                             (len(item[1]), item[0])):
        print("OPTION", {"keys": path, "move": move})


levels, path, error = arena.run_program("lf52", observe)
print("PROBE_RESULT", {"levels": levels, "moves": len(path),
                       "error": str(error)})

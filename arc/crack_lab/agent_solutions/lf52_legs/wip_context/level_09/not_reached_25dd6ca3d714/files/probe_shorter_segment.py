"""Shallow-root replay search for one exact key-segment substitution."""

import json
import os
import sys
from collections import deque

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import arr, safe_step


BOUNDARIES = (0, 8, 42, 87, 149, 238, 331, 476, 544)


def key(node):
    return int(node.levels_completed), arr(node.frame())[1:, :].tobytes()


def observe(env):
    level = int(os.environ.get("TARGET_LEVEL", "5"))
    wanted_macro = int(os.environ.get("TARGET_MACRO", "4"))
    max_states = int(os.environ.get("MAX_STATES", "10000"))
    max_depth = int(os.environ.get("MAX_DEPTH", "-1"))
    with open("checkpoint.json") as stream:
        admitted = json.load(stream)["final_path"]
    start, end = BOUNDARIES[level - 1], BOUNDARIES[level]
    for action in admitted[:start]:
        env.step(action)

    segment = admitted[start:end]
    index = 0
    macro_index = 0
    known = macro = None
    while index < len(segment):
        keys = []
        while index < len(segment) and not isinstance(segment[index], list):
            keys.append(segment[index])
            index += 1
        if index >= len(segment):
            break
        pair = tuple(tuple(action) for action in segment[index:index + 2])
        index += 2
        macro_index += 1
        if macro_index == wanted_macro:
            known, macro = tuple(keys), pair
            break
        for action in keys:
            safe_step(env, action)
        for action in pair:
            safe_step(env, action)
    if macro is None:
        raise ValueError("TARGET_MACRO is out of range")

    target = env.clone()
    for action in known + macro:
        safe_step(target, action)
    target_key = key(target)

    queue = deque([()])
    seen = {key(env)}
    solution = None
    while queue and len(seen) <= max_states:
        path = queue.popleft()
        node = env.clone()
        for action in path:
            safe_step(node, action)
        probe = node.clone()
        for action in macro:
            safe_step(probe, action)
        if key(probe) == target_key:
            solution = path
            break
        depth_limit = len(known) - 1 if max_depth < 0 else max_depth
        if len(path) >= depth_limit:
            continue
        for action in (1, 2, 3, 4):
            child = node.clone()
            safe_step(child, action)
            child_key = key(child)
            if child_key not in seen:
                seen.add(child_key)
                queue.append(path + (action,))

    print("SEARCH", {"level": level, "macro": wanted_macro,
                     "known": known, "known_len": len(known),
                     "states": len(seen), "shorter": solution,
                     "saving": 0 if solution is None else
                     len(known) - len(solution)})


levels, path, error = arena.run_program("lf52", observe)
print("PROBE_RESULT", {"levels": levels, "moves": len(path), "error": str(error)})

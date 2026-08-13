"""Find shorter drop-in key paths between admitted coordinate macros."""

import json
import os
import sys
from collections import deque

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import arr, safe_step


BOUNDARIES = (0, 8, 42, 87, 149, 238, 331, 476, 544)
KEYS = (1, 2, 3, 4)


def state_key(node):
    return int(node.levels_completed), arr(node.frame())[1:, :].tobytes()


def shorter_path(root, macro, target_key, admitted_length,
                 max_states=10000):
    if admitted_length == 0:
        return None, 0
    queue = deque([(root.clone(), ())])
    seen = {state_key(root)}
    while queue and len(seen) <= max_states:
        node, path = queue.popleft()
        probe = node.clone()
        for action in macro:
            safe_step(probe, action)
        if state_key(probe) == target_key:
            return path, len(seen)
        if len(path) >= admitted_length - 1:
            continue
        for action in KEYS:
            child = node.clone()
            safe_step(child, action)
            child_key = state_key(child)
            if child_key not in seen:
                seen.add(child_key)
                queue.append((child, path + (action,)))
    return None, len(seen)


def observe(env):
    level = int(os.environ.get("TARGET_LEVEL", "5"))
    with open("checkpoint.json") as stream:
        admitted = json.load(stream)["final_path"]
    start, end = BOUNDARIES[level - 1], BOUNDARIES[level]
    for action in admitted[:start]:
        env.step(action)

    current = env.clone()
    segment = admitted[start:end]
    index = 0
    macro_index = 0
    savings = 0
    replacements = []
    while index < len(segment):
        keys = []
        while index < len(segment) and not isinstance(segment[index], list):
            keys.append(segment[index])
            index += 1
        if index >= len(segment):
            break
        macro = tuple(tuple(action) for action in segment[index:index + 2])
        index += 2
        macro_index += 1

        start_node = current.clone()
        for action in keys:
            safe_step(current, action)
        for action in macro:
            safe_step(current, action)
        target = state_key(current)

        shorter, states = shorter_path(start_node, macro, target, len(keys))
        if shorter is not None and len(shorter) < len(keys):
            savings += len(keys) - len(shorter)
            replacements.append((macro_index, tuple(keys), shorter))
        print("SEGMENT", {"macro": macro_index, "old": tuple(keys),
                          "shorter": shorter, "states": states,
                          "saving": 0 if shorter is None else
                          len(keys) - len(shorter)})

    print("SUMMARY", {"level": level, "savings": savings,
                      "replacements": replacements})


levels, path, error = arena.run_program("lf52", observe)
print("PROBE_RESULT", {"levels": levels, "moves": len(path), "error": str(error)})

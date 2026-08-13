"""Replace level-5 carrier key runs by shortest paths to identical frames."""

from collections import deque
import json
import sys

sys.path.insert(0, "/private/tmp/gkm-submitted-protocol-lf52-20260805/arc/crack_lab")

import gkm_arena as arena

from perception import arr, safe_step


PREFIX_END = 149
LEVEL_END = 238
MAX_STATES_PER_RUN = 2000


def key(env):
    return int(env.levels_completed), arr(env.frame())[1:, :].tobytes()


def shortest_keys(root, target_key, depth_limit):
    if key(root) == target_key:
        return ()
    queue = deque([(root.clone(), ())])
    seen = {key(root)}
    while queue and len(seen) <= MAX_STATES_PER_RUN:
        node, path = queue.popleft()
        if len(path) >= depth_limit:
            continue
        for action in (1, 2, 3, 4):
            child = node.clone()
            safe_step(child, action)
            child_key = key(child)
            if child_key == target_key:
                return path + (action,)
            if child_key not in seen:
                seen.add(child_key)
                queue.append((child, path + (action,)))
    return None


def probe(env):
    with open("checkpoint.json") as handle:
        checkpoint = json.load(handle)
    full_path = checkpoint["final_path"]
    for action in full_path[:PREFIX_END]:
        env.step(action)
    base_level = int(env.levels_completed)
    original = tuple(full_path[PREFIX_END:LEVEL_END])
    node = env.clone()
    optimized = []
    index = 0
    run_number = 0
    while index < len(original):
        if isinstance(original[index], int):
            end = index
            while end < len(original) and isinstance(original[end], int):
                end += 1
            run = original[index:end]
            target = node.clone()
            for action in run:
                safe_step(target, action)
            replacement = shortest_keys(node, key(target), len(run) - 1)
            chosen = run if replacement is None else replacement
            for action in chosen:
                safe_step(node, action)
            assert key(node) == key(target)
            optimized.extend(chosen)
            run_number += 1
            print(
                "KEY_RUN", run_number, len(run), len(chosen), tuple(chosen),
                flush=True,
            )
            index = end
        else:
            macro = original[index:index + 2]
            for action in macro:
                safe_step(node, action)
            optimized.extend(macro)
            index += 2
    print("LEVEL5_SHORTENED", len(original), len(optimized), node.levels_completed)
    print("LEVEL5_ACTIONS", json.dumps(optimized, separators=(",", ":")))
    print("BASE_LEVEL", base_level)


levels, path, error = arena.run_program("lf52", probe)
print("PROBE_RESULT", levels, len(path), error)

"""Search carrier peg levels with every advertised non-coordinate action."""

from collections import deque
import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from legs import _carrier_capture_macros, play_action
from perception import arr


TARGET_LEVEL = int(os.environ.get("TARGET_LEVEL", "2"))
BOUNDARIES = (0, 8, 42, 87)
MAX_STATES = int(os.environ.get("MAX_STATES", "10000"))
MAX_DEPTH = int(os.environ.get("MAX_DEPTH", "40"))


def key(node):
    return int(node.levels_completed), arr(node.frame())[1:, :].tobytes()


def search(root):
    base = int(root.levels_completed)
    queue = deque([(root.clone(), ())])
    seen = {key(root)}
    while queue and len(seen) < MAX_STATES:
        node, path = queue.popleft()
        if len(path) >= MAX_DEPTH:
            continue
        macros = tuple((action,) for action in (1, 2, 3, 4, 7))
        macros += _carrier_capture_macros(node.frame())
        for macro in macros:
            child = node.clone()
            for action in macro:
                play_action(child, action)
            child_path = path + (macro,)
            if child.levels_completed > base:
                return child_path, len(seen)
            child_key = key(child)
            if child_key not in seen:
                seen.add(child_key)
                queue.append((child, child_path))
    return None, len(seen)


def flatten(path):
    return [action for macro in path for action in macro]


def probe(env):
    with open("checkpoint.json") as stream:
        campaign = json.load(stream)["final_path"]
    entry = BOUNDARIES[TARGET_LEVEL - 1]
    for action in campaign[:entry]:
        play_action(env, tuple(action) if isinstance(action, list) else action)
    result, states = search(env)
    actions = None if result is None else flatten(result)
    verified = None
    if actions is not None:
        node = env.clone()
        for action in actions:
            play_action(node, action)
        verified = int(node.levels_completed)
    print("ALL_KEYS", {"target": TARGET_LEVEL, "states": states,
                       "actions": None if actions is None else len(actions),
                       "macro_depth": None if result is None else len(result),
                       "verified": verified, "path": actions}, flush=True)


levels, path, error = arena.run_program("lf52", probe)
print("HARNESS", {"levels": levels, "moves": len(path), "error": str(error)})

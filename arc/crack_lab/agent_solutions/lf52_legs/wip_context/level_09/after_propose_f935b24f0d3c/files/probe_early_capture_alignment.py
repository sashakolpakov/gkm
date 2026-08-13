"""Find shortest all-key alignment to the next carrier peg capture."""

from collections import deque
import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from legs import _carrier_capture_macros
from perception import arr, safe_step


TARGET_LEVEL = int(os.environ.get("TARGET_LEVEL", "2"))
LOCAL_CONTEXT = int(os.environ.get("LOCAL_CONTEXT", "14"))
EXTRA_ACTIONS = json.loads(os.environ.get("EXTRA_ACTIONS", "[]"))
INCLUDE7 = os.environ.get("INCLUDE7", "1") == "1"
MAX_STATES = int(os.environ.get("MAX_STATES", "1200"))
MAX_DEPTH = int(os.environ.get("MAX_DEPTH", "24"))
BOUNDARIES = (0, 8, 42, 87)


def play(node, action):
    safe_step(node, tuple(action) if isinstance(action, list) else action)


def key(node):
    return int(node.levels_completed), arr(node.frame())[1:, :].tobytes()


def search(root):
    actions = (1, 2, 3, 4, 7) if INCLUDE7 else (1, 2, 3, 4)
    queue = deque([(root.clone(), ())])
    seen = {key(root)}
    while queue and len(seen) <= MAX_STATES:
        node, path = queue.popleft()
        macros = _carrier_capture_macros(node.frame())
        if macros:
            return path, macros, len(seen)
        if len(path) >= MAX_DEPTH:
            continue
        for action in actions:
            child = node.clone()
            play(child, action)
            child_key = key(child)
            if child_key in seen:
                continue
            seen.add(child_key)
            queue.append((child, path + (action,)))
    return None, (), len(seen)


def probe(env):
    with open("checkpoint.json") as stream:
        campaign = json.load(stream)["final_path"]
    start = BOUNDARIES[TARGET_LEVEL - 1]
    for action in campaign[:start + LOCAL_CONTEXT]:
        play(env, action)
    for action in EXTRA_ACTIONS:
        play(env, action)
    path, macros, states = search(env)
    print("CAPTURE_ALIGNMENT", {"level": TARGET_LEVEL,
          "context": LOCAL_CONTEXT, "include7": INCLUDE7,
          "states": states, "cost": None if path is None else len(path),
          "path": path, "macros": macros}, flush=True)


levels, path, error = arena.run_program("lf52", probe)
print("HARNESS", {"levels": levels, "moves": len(path), "error": str(error)})

"""Compact key-only orbit at pristine level-7 entry."""

from collections import deque
import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from legs import _bridge_carrier_state
from perception import arr, safe_step


ENTRY = 331
MAX_STATES = int(os.environ.get("MAX_STATES", "180"))
MAX_DEPTH = int(os.environ.get("MAX_DEPTH", "12"))


def frame_key(node):
    return arr(node.frame()).tobytes()


def compact(node):
    state = _bridge_carrier_state(node.frame())
    return tuple(sorted(state[2])), tuple(sorted(state[1]))


def probe(env):
    with open("checkpoint.json") as stream:
        campaign = json.load(stream)["final_path"]
    for action in campaign[:ENTRY]:
        safe_step(env, action)
    root = env.clone()
    start = frame_key(root)
    queue = deque([(root, (), start)])
    seen = {start}
    depth_counts = {}
    changed = inert = 0
    examples = []
    while queue and len(seen) < MAX_STATES:
        node, path, parent_key = queue.popleft()
        depth_counts[len(path)] = depth_counts.get(len(path), 0) + 1
        if len(path) >= MAX_DEPTH:
            continue
        for action in (1, 2, 3, 4):
            child = node.clone()
            safe_step(child, action)
            child_key = frame_key(child)
            if child_key == parent_key:
                inert += 1
            else:
                changed += 1
            if child_key not in seen:
                seen.add(child_key)
                child_path = path + (action,)
                queue.append((child, child_path, child_key))
                if len(examples) < 24:
                    examples.append((child_path, compact(child)))
    print("KEY_ORBIT", {
        "states": len(seen), "queued": len(queue),
        "depth_counts": depth_counts,
        "changed_edges": changed, "inert_edges": inert,
        "examples": examples,
    })


levels, path, error = arena.run_program("lf52", probe)
print("HARNESS", {"levels": levels, "moves": len(path), "error": str(error)})

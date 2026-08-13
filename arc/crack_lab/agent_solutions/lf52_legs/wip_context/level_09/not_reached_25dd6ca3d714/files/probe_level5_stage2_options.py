"""Enumerate shortest key alignments to level-5 stage-two peg moves."""

import json
import sys
from collections import deque
sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from legs import _bridge_carrier_moves, _bridge_carrier_state
from perception import arr, safe_step


STAGE2_PREFIX = 166


def observe(env):
    with open("checkpoint.json") as stream:
        path = json.load(stream)["final_path"]
    for action in path[:STAGE2_PREFIX]:
        env.step(action)
    root = env.clone()
    frame_key = lambda candidate: arr(candidate.frame())[1:, :].tobytes()
    queue = deque([(root, ())])
    seen = {frame_key(root)}
    options = {}
    max_states, max_depth = 1000, 16
    while queue and len(seen) <= max_states:
        node, key_path = queue.popleft()
        state = frame_key(node)
        for move in _bridge_carrier_moves(node.frame()):
            signature = (move, state)
            options.setdefault(signature, key_path)
        if len(key_path) >= max_depth:
            continue
        for action in (1, 2, 3, 4):
            child = node.clone()
            safe_step(child, action)
            child_state = frame_key(child)
            if child_state not in seen:
                seen.add(child_state)
                queue.append((child, key_path + (action,)))

    compact = sorted(
        ((move, path) for (move, _), path in options.items()),
        key=lambda item: (item[0][0] != "capture", len(item[1]), item),
    )
    print("SEARCH", {"states": len(seen), "queued": len(queue),
                     "options": len(compact)})
    for move, key_path in compact:
        print("OPTION", {"move": move, "keys": key_path,
                         "cost": len(key_path) + 2})


levels, path, error = arena.run_program("lf52", observe)
print("PROBE_RESULT", {"levels": levels, "moves": len(path), "error": str(error)})

"""Bounded symbolic search over level-9 key-controlled carrier states."""

import json
import sys
from collections import deque

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import arr, connected_components, safe_step


KEYS = (1, 2, 3, 4, 7)


def playfield_key(env):
    return arr(env.frame())[1:, :].tobytes()


def carrier_positions(frame):
    return tuple(
        blob.top_left for blob in connected_components(frame, colors=(12,))
        if blob.size == (4, 4) and blob.area == 16
    )


def peg_bridge_counts(frame):
    pegs = sum(
        blob.area == 12 and blob.size == (4, 4)
        for blob in connected_components(frame, colors=(14,))
    )
    small_bridges = sum(
        blob.area == 12 and blob.size == (4, 4)
        for blob in connected_components(frame, colors=(9,))
    )
    return pegs, small_bridges


def observe(env):
    with open("checkpoint.json") as stream:
        prefix = json.load(stream)["final_path"]
    for action in prefix:
        env.step(action)

    root = env.clone()
    queue = deque([(root, ())])
    seen = {playfield_key(root)}
    records = []
    max_states, max_depth = 500, 24
    while queue and len(seen) <= max_states:
        node, path = queue.popleft()
        records.append((path, carrier_positions(node.frame()),
                        peg_bridge_counts(node.frame()), node.levels_completed))
        if len(path) >= max_depth:
            continue
        for action in KEYS:
            child = node.clone()
            safe_step(child, action)
            key = playfield_key(child)
            if key not in seen:
                seen.add(key)
                queue.append((child, path + (action,)))

    print("KEY_SEARCH", {"states": len(seen), "expanded": len(records),
                         "queued": len(queue), "max_depth": max_depth})
    for record in records:
        print("STATE", record)


levels, path, error = arena.run_program("lf52", observe)
print("PROBE_RESULT", {"levels": levels, "moves": len(path), "error": str(error)})

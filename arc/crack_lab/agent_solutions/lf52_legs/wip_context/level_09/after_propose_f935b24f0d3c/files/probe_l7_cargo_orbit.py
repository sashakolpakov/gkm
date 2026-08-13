"""Map key-only cargo positions for a reproduced level-7 branch."""

from collections import deque
import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from legs import _movable_bridge_board
from perception import safe_step


CONTEXT = int(os.environ.get("CONTEXT_ACTIONS", "109"))
EXTRA = json.loads(os.environ.get("EXTRA_ACTIONS", "[]"))
MAX_STATES = int(os.environ.get("MAX_STATES", "300"))
MAX_DEPTH = int(os.environ.get("MAX_DEPTH", "30"))
TRACE_KEYS = json.loads(os.environ.get("TRACE_KEYS", "[]"))


def play(node, action):
    safe_step(node, tuple(action) if isinstance(action, list) else action)


def state(node):
    board = _movable_bridge_board(node.frame())
    return tuple(tuple(sorted(part)) for part in board)


def probe(env):
    with open("checkpoint.json") as stream:
        campaign = json.load(stream)["final_path"]
    with open("level7_greedy_macro_candidate.json") as stream:
        candidate = json.load(stream)
    for action in campaign[:331]:
        play(env, action)
    for action in candidate[:CONTEXT]:
        play(env, action)
    for action in EXTRA:
        play(env, action)
    root = env.clone()
    root_state = state(root)
    if TRACE_KEYS:
        rows = []
        node = root.clone()
        for index, action in enumerate(TRACE_KEYS, 1):
            play(node, action)
            rows.append((index, action, state(node)))
        print("CARGO_TRACE", {"root": root_state, "rows": rows}, flush=True)
        return
    queue = deque([(root, ())])
    seen = {root_state}
    positions = {root_state[3]: ()}
    while queue and len(seen) <= MAX_STATES:
        node, path = queue.popleft()
        if len(path) >= MAX_DEPTH:
            continue
        for action in (1, 2, 3, 4):
            child = node.clone()
            play(child, action)
            child_state = state(child)
            if child_state in seen:
                continue
            seen.add(child_state)
            child_path = path + (action,)
            queue.append((child, child_path))
            positions.setdefault(child_state[3], child_path)
    print("CARGO_ORBIT", {"root": root_state, "states": len(seen),
                          "queued": len(queue), "positions": len(positions)},
          flush=True)
    for pegs, path in sorted(positions.items(), key=lambda x: (len(x[1]), x[1])):
        print("POSITION", {"keys": path, "pegs": pegs}, flush=True)


levels, path, error = arena.run_program("lf52", probe)
print("HARNESS", {"levels": levels, "moves": len(path), "error": str(error)})

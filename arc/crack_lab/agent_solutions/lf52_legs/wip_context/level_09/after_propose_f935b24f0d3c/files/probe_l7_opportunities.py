"""Enumerate coordinate moves reachable by key-only motion at a level-7 boundary."""

from collections import deque
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from legs import _bridge_carrier_state, _movable_bridge_board
from perception import safe_step


ENTRY = int(os.environ.get("ENTRY_INDEX", "331"))
CONTEXT = int(os.environ.get("CONTEXT_ACTIONS", "0"))
CANDIDATE = os.environ.get(
    "CANDIDATE", "level7_greedy_macro_candidate.json"
)
EXTRA_ACTIONS = json.loads(os.environ.get("EXTRA_ACTIONS", "[]"))
MAX_DEPTH = int(os.environ.get("KEY_MAX_DEPTH", "18"))
MAX_STATES = int(os.environ.get("KEY_MAX_STATES", "180"))
FAST_MOVES = os.environ.get("FAST_MOVES", "1") == "1"
STATE_KEY_MODE = os.environ.get("STATE_KEY", "raw")
KEY_ACTIONS = ((1, 2, 3, 4, 7) if os.environ.get("INCLUDE7") == "1"
               else (1, 2, 3, 4))


def play(node, action):
    safe_step(node, tuple(action) if isinstance(action, list) else action)


def physical_key(node):
    if STATE_KEY_MODE in ("compact", "simple"):
        movable = _movable_bridge_board(node.frame())
        movable_key = tuple(tuple(sorted(part)) for part in movable)
        if STATE_KEY_MODE == "simple":
            return int(node.levels_completed), movable_key
        fixed = _bridge_carrier_state(node.frame())
        return (int(node.levels_completed),
                movable_key, fixed[:5])
    frame = np.asarray(node.frame()).copy()
    frame[0, :] = 0
    return int(node.levels_completed), frame.tobytes()


def board(frame):
    slots, carriers, bridges, pegs = _movable_bridge_board(frame)
    array = np.asarray(frame)
    windows = np.lib.stride_tricks.sliding_window_view(array, (4, 4))
    counts = {
        color: np.count_nonzero(windows == color, axis=(-1, -2))
        for color in (1, 8, 9, 12, 14)
    }

    def positions(mask):
        rows, cols = np.where(mask)
        return set(zip(map(int, rows), map(int, cols)))

    slots |= positions(counts[1] == 16)
    carriers |= positions(counts[12] == 16)
    bridges |= positions(counts[8] >= 12) | positions(counts[9] >= 12)
    pegs |= positions(counts[14] >= 12)
    fixed_state = _bridge_carrier_state(frame)
    slots |= set(fixed_state[0])
    carriers |= set(fixed_state[2])
    pegs |= set(fixed_state[1])
    return set(slots), set(carriers), set(bridges), set(pegs), set(fixed_state[3])


def visible_moves(node):
    slots, carriers, bridges, pegs, fixed = board(node.frame())
    destinations = slots | carriers
    occupied = bridges | pegs | fixed
    result = []
    for kind, pieces in (("bridge", bridges), ("peg", pegs)):
        for source in sorted(pieces):
            for dr, dc in ((-6, 0), (6, 0), (0, -6), (0, 6)):
                midpoint = (source[0] + dr, source[1] + dc)
                destination = (source[0] + 2 * dr, source[1] + 2 * dc)
                if (midpoint in occupied and destination in destinations
                        and destination not in occupied):
                    result.append((kind, source, destination))
    return tuple(result)


def move_actions(move):
    _, source, destination = move
    return ((6, source[1] + 1, source[0] + 1),
            (6, destination[1] + 1, destination[0] + 1))


def valid_move(before, after, move):
    if after.levels_completed > before.levels_completed:
        return True
    kind, source, destination = move
    _, _, bridges, pegs, _ = board(after.frame())
    pieces = bridges if kind == "bridge" else pegs
    if source not in pieces and destination in pieces:
        return True
    array = np.asarray(after.frame())
    return (physical_key(before) != physical_key(after)
            and not np.isin(array, (2, 3)).any())


def compact(node):
    _, carriers, bridges, pegs, _ = board(node.frame())
    return {"pegs": sorted(pegs), "bridges": sorted(bridges),
            "carriers": sorted(carriers), "level": int(node.levels_completed)}


def probe(env):
    with open("checkpoint.json") as stream:
        campaign = json.load(stream)["final_path"]
    with open(CANDIDATE) as stream:
        candidate = json.load(stream)
    for action in campaign[:ENTRY]:
        play(env, action)
    for action in candidate[:CONTEXT]:
        play(env, action)
    for action in EXTRA_ACTIONS:
        play(env, action)
    root = env.clone()
    queue = deque([(root, ())])
    seen = {physical_key(root)}
    children = {}
    predicted = {}
    while queue and len(seen) <= MAX_STATES:
        node, path = queue.popleft()
        for move in visible_moves(node):
            if FAST_MOVES:
                previous = predicted.get(move)
                if previous is None or len(path) < len(previous[0]):
                    predicted[move] = (path, node.clone())
                continue
            child = node.clone()
            for action in move_actions(move):
                play(child, action)
            if not valid_move(node, child, move):
                continue
            key = physical_key(child)
            value = (path, move, compact(child))
            if key not in children or len(path) < len(children[key][0]):
                children[key] = value
        if len(path) >= MAX_DEPTH:
            continue
        for action in KEY_ACTIONS:
            child = node.clone()
            play(child, action)
            key = physical_key(child)
            if key in seen:
                continue
            seen.add(key)
            queue.append((child, path + (action,)))
    if FAST_MOVES:
        for move, (path, node) in predicted.items():
            child = node.clone()
            for action in move_actions(move):
                play(child, action)
            if not valid_move(node, child, move):
                continue
            children[move] = (path, move, compact(child))
    values = sorted(children.values(), key=lambda x: (len(x[0]), x[0], x[1]))
    print("L7_OPPORTUNITIES", {
        "context": CONTEXT, "root": compact(root),
        "key_states": len(seen), "queued": len(queue),
        "children": len(values),
    }, flush=True)
    for path, move, summary in values:
        print("MOVE", {"keys": path, "key_cost": len(path),
                       "move": move, "child": summary}, flush=True)


levels, path, error = arena.run_program("lf52", probe)
print("HARNESS", {"levels": levels, "moves": len(path), "error": str(error)})

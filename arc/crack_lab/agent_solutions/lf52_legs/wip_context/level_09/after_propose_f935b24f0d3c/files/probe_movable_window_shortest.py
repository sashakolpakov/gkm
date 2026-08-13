"""Find a cheaper exact-frame path across one movable-bridge window."""

from heapq import heappop, heappush
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from legs import _bridge_carrier_state, _movable_bridge_board
from perception import safe_step


LEVEL_ENTRY = int(os.environ.get("WINDOW_LEVEL_ENTRY", "238"))
WINDOW_START = int(os.environ.get("WINDOW_START", "58"))
WINDOW_END = int(os.environ.get("WINDOW_END", "70"))
MAX_COST = int(os.environ.get("WINDOW_MAX_COST", str(WINDOW_END - WINDOW_START - 1)))
MAX_STATES = int(os.environ.get("WINDOW_MAX_STATES", "1200"))
CANDIDATE_FILE = os.environ.get(
    "WINDOW_CANDIDATE", "level6_greedy_macro_candidate.json"
)


def play(env, action):
    safe_step(env, tuple(action) if isinstance(action, list) else action)


def move_actions(move):
    _, source, destination = move
    return (
        (6, source[1] + 1, source[0] + 1),
        (6, destination[1] + 1, destination[0] + 1),
    )


def physical_key(env):
    frame = np.asarray(env.frame()).copy()
    frame[0, :] = 0
    return int(env.levels_completed), frame.tobytes()


def board(frame):
    """Include both color-8 and color-9 movable bridge variants."""
    slots, carriers, bridges, pegs = _movable_bridge_board(frame)
    array = np.asarray(frame)
    windows = np.lib.stride_tricks.sliding_window_view(array, (4, 4))
    count1 = np.count_nonzero(windows == 1, axis=(-1, -2))
    count9 = np.count_nonzero(windows == 9, axis=(-1, -2))
    rows, cols = np.where((count9 >= 12) & ((count9 + count1) == 16))
    bridges = set(bridges) | set(zip(map(int, rows), map(int, cols)))
    bridge_state = _bridge_carrier_state(frame)
    slots = set(slots) | set(bridge_state[0])
    carriers = set(carriers) | set(bridge_state[2])
    pegs = set(pegs) | set(bridge_state[1])
    return slots, carriers, bridges, pegs


def visible_moves(env):
    slots, carriers, bridges, pegs = board(env.frame())
    bridge_state = _bridge_carrier_state(env.frame())
    destinations = set(slots) | set(carriers) | set(bridge_state[0]) | set(bridge_state[2])
    fixed = set(bridge_state[3])
    occupied = set(bridges) | set(pegs)
    result = []
    for kind, pieces in (("bridge", bridges), ("peg", pegs)):
        for source in sorted(pieces):
            for dr, dc in ((-6, 0), (6, 0), (0, -6), (0, 6)):
                midpoint = (source[0] + dr, source[1] + dc)
                destination = (source[0] + 2 * dr, source[1] + 2 * dc)
                if (
                    midpoint in occupied | fixed
                    and destination in destinations
                    and destination not in occupied | fixed
                ):
                    result.append((kind, source, destination))
    return tuple(result)


def apply_move(node, move, target_key):
    child = node.clone()
    for action in move_actions(move):
        play(child, action)
    if physical_key(child) == target_key:
        return child
    kind, source, destination = move
    _, _, bridges, pegs = board(child.frame())
    pieces = bridges if kind == "bridge" else pegs
    return child if source not in pieces and destination in pieces else None


def search(root, target_key):
    serial = 0
    queue = [(0, serial, root.clone(), ())]
    best = {physical_key(root): 0}
    expanded = 0
    while queue and expanded < MAX_STATES:
        cost, _, node, path = heappop(queue)
        if cost != best.get(physical_key(node)):
            continue
        expanded += 1
        if physical_key(node) == target_key:
            return path, expanded, len(best)
        if cost + 1 <= MAX_COST:
            for action in (1, 2, 3, 4):
                child = node.clone()
                play(child, action)
                child_cost = cost + 1
                key = physical_key(child)
                if child_cost < best.get(key, 10 ** 9):
                    best[key] = child_cost
                    serial += 1
                    heappush(queue, (
                        child_cost, serial, child, path + (action,)
                    ))
        if cost + 2 <= MAX_COST:
            for move in visible_moves(node):
                child = apply_move(node, move, target_key)
                if child is None:
                    continue
                child_cost = cost + 2
                child_path = path + move_actions(move)
                if physical_key(child) == target_key:
                    return child_path, expanded, len(best)
                key = physical_key(child)
                if child_cost < best.get(key, 10 ** 9):
                    best[key] = child_cost
                    serial += 1
                    heappush(queue, (
                        child_cost, serial, child, child_path
                    ))
    return None, expanded, len(best)


def probe(env):
    with open("checkpoint.json") as stream:
        campaign = json.load(stream)["final_path"]
    with open(CANDIDATE_FILE) as stream:
        candidate = json.load(stream)
    for action in campaign[:LEVEL_ENTRY]:
        play(env, action)
    for action in candidate[:WINDOW_START]:
        play(env, action)
    root = env.clone()
    target = root.clone()
    for action in candidate[WINDOW_START:WINDOW_END]:
        play(target, action)
    target_key = physical_key(target)
    result = search(root, target_key)
    print("WINDOW", {
        "candidate": CANDIDATE_FILE,
        "range": (WINDOW_START, WINDOW_END),
        "known_cost": WINDOW_END - WINDOW_START,
        "max_cost": MAX_COST,
        "path": result[0],
        "cost": None if result[0] is None else len(result[0]),
        "expanded": result[1],
        "seen": result[2],
    }, flush=True)


levels, path, error = arena.run_program("lf52", probe)
print("HARNESS", {"levels": levels, "moves": len(path), "error": str(error)})

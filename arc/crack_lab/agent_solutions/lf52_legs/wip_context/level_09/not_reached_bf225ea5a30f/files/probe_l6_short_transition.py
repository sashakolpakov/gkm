"""Search short continuations from alternate level-6 carrier arrivals."""

from heapq import heappop, heappush
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from legs import _bridge_carrier_state, _movable_bridge_board
from perception import safe_step


LEVEL_ENTRY = 238
LOCAL_PREFIX = 22
MAX_STATES = int(os.environ.get("L6_ALT_MAX_STATES", "1200"))
MAX_REMAINDER = int(os.environ.get("L6_ALT_MAX_COST", "11"))

BRANCHES = {
    "bridge_34_up": (
        (4, 4, 4, 4, 4, 4, 1, 1),
        ("bridge", (30, 34), (18, 34)),
    ),
    "peg_28_up": (
        (4, 4, 4, 4, 4, 4, 4, 1, 1),
        ("peg", (30, 28), (18, 28)),
    ),
    "bridge_28_up": (
        (4, 4, 4, 4, 4, 4, 1, 4, 1),
        ("bridge", (30, 28), (18, 28)),
    ),
}


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
    return frame.tobytes()


def phase_goal(env):
    return len(_movable_bridge_board(env.frame())[3]) >= 4


def visible_moves(env):
    slots, carriers, bridges, pegs = _movable_bridge_board(env.frame())
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


def valid_move(before, after, move, goal_fn=phase_goal):
    if goal_fn(after):
        return True
    kind, source, destination = move
    _, _, bridges, pegs = _movable_bridge_board(after.frame())
    pieces = bridges if kind == "bridge" else pegs
    return source not in pieces and destination in pieces


def apply_move(node, move, goal_fn=phase_goal):
    child = node.clone()
    for action in move_actions(move):
        play(child, action)
    return child if valid_move(node, child, move, goal_fn) else None


def search(root, goal_fn):
    serial = 0
    queue = [(0, serial, root.clone(), ())]
    best = {physical_key(root): 0}
    expanded = 0
    while queue and expanded < MAX_STATES:
        cost, _, node, path = heappop(queue)
        if cost != best.get(physical_key(node)):
            continue
        expanded += 1
        if goal_fn(node):
            return path, expanded, len(best)
        if cost + 1 <= MAX_REMAINDER:
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
        if cost + 2 <= MAX_REMAINDER:
            for move in visible_moves(node):
                child = apply_move(node, move, goal_fn)
                if child is None:
                    continue
                child_cost = cost + 2
                child_path = path + move_actions(move)
                if goal_fn(child):
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
    branch_name = os.environ.get("L6_ALT_BRANCH", "bridge_34_up")
    keys, move = BRANCHES[branch_name]
    with open("checkpoint.json") as stream:
        campaign = json.load(stream)["final_path"]
    with open("level6_greedy_macro_candidate.json") as stream:
        candidate = json.load(stream)
    for action in campaign[:LEVEL_ENTRY]:
        play(env, action)
    for action in candidate[:LOCAL_PREFIX]:
        play(env, action)
    target = env.clone()
    for action in candidate[LOCAL_PREFIX:44]:
        play(target, action)
    target_key = physical_key(target)
    goal_fn = lambda node: physical_key(node) == target_key
    root = env.clone()
    for action in keys:
        play(root, action)
    child = apply_move(root, move, goal_fn)
    if child is None:
        raise RuntimeError(f"branch move is not valid: {move}")
    result = search(child, goal_fn)
    print("BRANCH", {
        "name": branch_name,
        "entry_cost": len(keys) + 2,
        "move": move,
        "board": tuple(map(sorted, _movable_bridge_board(child.frame()))),
    }, flush=True)
    print("SEARCH", {
        "remainder_limit": MAX_REMAINDER,
        "path": result[0],
        "cost": None if result[0] is None else len(result[0]),
        "expanded": result[1],
        "seen": result[2],
        "known_transition_cost": 22,
    }, flush=True)


levels, path, error = arena.run_program("lf52", probe)
print("HARNESS", {"levels": levels, "moves": len(path), "error": str(error)})

"""Enumerate validated cargo moves reachable through carrier-only key motion."""

from collections import deque
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from legs import _bridge_carrier_state


CONTEXT_INDEX = int(os.environ.get("CONTEXT_INDEX", "28"))
MAX_DEPTH = int(os.environ.get("CARRIER_MAX_DEPTH", "15"))
MAX_STATES = int(os.environ.get("CARRIER_MAX_STATES", "300"))
EXTRA_ACTIONS = json.loads(os.environ.get("EXTRA_ACTIONS", "[]"))


def play(env, action):
    if isinstance(action, list):
        env.step(*action)
    else:
        env.step(action)


def physical_key(env):
    frame = np.asarray(env.frame()).copy()
    frame[0, :] = 0
    return frame.tobytes()


def lattice(frame):
    array = np.asarray(frame)
    windows = np.lib.stride_tricks.sliding_window_view(array, (4, 4))
    counts = {
        color: np.count_nonzero(windows == color, axis=(-1, -2))
        for color in (1, 9, 12, 14, 15)
    }

    def positions(mask):
        rows, cols = np.where(mask)
        return frozenset(zip(map(int, rows), map(int, cols)))

    holes = positions(counts[1] == 16)
    movable_bridges = positions(
        (counts[9] >= 12) & ((counts[9] + counts[1]) == 16)
    )
    pegs = positions(counts[14] >= 12)
    carriers = positions(counts[12] == 16)
    supports = positions(counts[15] >= 12)
    state = _bridge_carrier_state(frame)
    holes |= state[0]
    pegs |= state[1]
    carriers |= state[2]
    supports |= state[3]
    return holes, movable_bridges, pegs, carriers, supports


def visible_moves(frame):
    holes, bridges, pegs, carriers, supports = lattice(frame)
    destinations = holes | carriers
    occupied = bridges | pegs | supports
    moves = []
    for kind, pieces in (("bridge", bridges), ("peg", pegs)):
        for source in sorted(pieces):
            for dr, dc in ((-6, 0), (6, 0), (0, -6), (0, 6)):
                midpoint = (source[0] + dr, source[1] + dc)
                destination = (source[0] + 2 * dr, source[1] + 2 * dc)
                if midpoint in occupied and destination in destinations:
                    moves.append((kind, source, destination))
    return tuple(moves)


def move_actions(move):
    _, source, destination = move
    return (
        [6, source[1] + 1, source[0] + 1],
        [6, destination[1] + 1, destination[0] + 1],
    )


def valid_move(before, after, move):
    if after.levels_completed > before.levels_completed:
        return True
    kind, source, destination = move
    color = 9 if kind == "bridge" else 14
    array = np.asarray(after.frame())

    def count(position):
        row, col = position
        return int(np.count_nonzero(
            array[row:row + 4, col:col + 4] == color
        ))

    return count(source) < 12 and count(destination) >= 12


def summary(env):
    _, movable, pegs, carriers, supports = lattice(env.frame())
    return {
        "pegs": sorted(pegs),
        "movable": sorted(movable),
        "carriers": sorted(carriers),
        "supports": sorted(supports),
        "level": env.levels_completed,
    }


def explore(root):
    def reconstruct(path):
        node = root.clone()
        for action in path:
            node.step(action)
        return node

    queue = deque([()])
    seen = {physical_key(root)}
    opportunities = {}
    while queue and len(seen) <= MAX_STATES:
        path = queue.popleft()
        node = reconstruct(path)
        for move in visible_moves(node.frame()):
            child = node.clone()
            for action in move_actions(move):
                play(child, action)
            if not valid_move(node, child, move):
                continue
            child_key = physical_key(child)
            candidate = (path, move, summary(child))
            previous = opportunities.get(child_key)
            if previous is None or len(path) < len(previous[0]):
                opportunities[child_key] = candidate
        if len(path) >= MAX_DEPTH:
            continue
        for action in (1, 2, 3, 4):
            child_path = path + (action,)
            child = reconstruct(child_path)
            key = physical_key(child)
            if key in seen:
                continue
            seen.add(key)
            queue.append(child_path)
    return seen, opportunities


def probe(env):
    with open("checkpoint.json") as checkpoint_file:
        prefix = json.load(checkpoint_file)["final_path"]
    with open("level9_candidate_102.json") as candidate_file:
        candidate = json.load(candidate_file)
    for action in prefix:
        play(env, action)
    node = env.clone()
    for action in candidate[:CONTEXT_INDEX]:
        play(node, action)
    for action in EXTRA_ACTIONS:
        play(node, action)
    seen, opportunities = explore(node)
    print("CONTEXT", {
        "index": CONTEXT_INDEX,
        "extra_cost": len(EXTRA_ACTIONS),
        "summary": summary(node),
        "carrier_states": len(seen),
        "opportunities": len(opportunities),
    }, flush=True)
    for path, move, child_summary in sorted(
        opportunities.values(),
        key=lambda value: (len(value[0]), value[0], value[1]),
    ):
        print("OPPORTUNITY", {
            "keys": path,
            "key_cost": len(path),
            "move": move,
            "child": child_summary,
        }, flush=True)


levels, path, error = arena.run_program("lf52", probe)
print("HARNESS", {"levels": levels, "moves": len(path), "error": str(error) if error else None})

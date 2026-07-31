"""Enumerate level-7 cargo moves reachable through carrier-only key motion."""

from collections import deque
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from legs import _movable_bridge_board


MAX_DEPTH = int(os.environ.get("CARRIER_MAX_DEPTH", "18"))
MAX_STATES = int(os.environ.get("CARRIER_MAX_STATES", "500"))
EXTRA_ACTIONS = json.loads(os.environ.get("EXTRA_ACTIONS", "[]"))
EXTRA_FILE = os.environ.get("EXTRA_FILE")
EXTRA_LIMIT = int(os.environ.get("EXTRA_LIMIT", "0"))
FORWARD_ONLY = os.environ.get("FORWARD_ONLY") == "1"
KEY_TEST = os.environ.get("KEY_TEST") == "1"


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
        for color in (1, 8, 9, 12, 14, 15)
    }

    def positions(mask):
        rows, cols = np.where(mask)
        return frozenset(zip(map(int, rows), map(int, cols)))

    holes = positions(counts[1] == 16)
    bridges = positions(counts[8] >= 12)
    pegs = positions(counts[14] >= 12)
    carriers = positions(counts[12] == 16)
    supports = set()
    supports |= {
        (row + 1, col)
        for row, col in positions(counts[15] >= 12)
        if row < 60
    }
    parsed = _movable_bridge_board(frame)
    holes |= parsed[0]
    carriers |= parsed[1]
    bridges |= parsed[2]
    pegs |= parsed[3]
    return holes, bridges, pegs, carriers, frozenset(supports)


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
                if (
                    FORWARD_ONLY
                    and source[0] == destination[0]
                    and destination[1] < source[1]
                ):
                    continue
                if (
                    midpoint not in occupied
                    or destination not in destinations
                ):
                    continue
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
    color = 8 if kind == "bridge" else 14
    array = np.asarray(after.frame())

    def count(position):
        row, col = position
        return int(np.count_nonzero(
            array[row:row + 4, col:col + 4] == color
        ))

    return count(source) < 12 and count(destination) >= 12


def summary(env):
    holes, bridges, pegs, carriers, supports = lattice(env.frame())
    return {
        "holes": len(holes),
        "bridges": sorted(bridges),
        "pegs": sorted(pegs),
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
            key = physical_key(child)
            candidate = (path, move, summary(child))
            previous = opportunities.get(key)
            if previous is None or len(path) < len(previous[0]):
                opportunities[key] = candidate
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
        campaign = json.load(checkpoint_file)["final_path"]
    for action in campaign[:331]:
        play(env, action)
    node = env.clone()
    file_actions = []
    if EXTRA_FILE:
        with open(EXTRA_FILE) as extra_file:
            file_actions = json.load(extra_file)
        if EXTRA_LIMIT:
            file_actions = file_actions[:EXTRA_LIMIT]
    for action in file_actions:
        play(node, action)
    for action in EXTRA_ACTIONS:
        play(node, action)
    if KEY_TEST:
        before_key = physical_key(node)
        print("KEY_CONTEXT", {
            "extra_cost": len(EXTRA_ACTIONS),
            "file_cost": len(file_actions),
            "summary": summary(node),
        }, flush=True)
        for action in (1, 2, 3, 4):
            child = node.clone()
            child.step(action)
            print("KEY", {
                "action": action,
                "changed": physical_key(child) != before_key,
                "summary": summary(child),
            }, flush=True)
        return
    seen, opportunities = explore(node)
    print("CONTEXT", {
        "extra_cost": len(EXTRA_ACTIONS),
        "file_cost": len(file_actions),
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


if __name__ == "__main__":
    levels, path, error = arena.run_program("lf52", probe)
    print("HARNESS", {
        "levels": levels,
        "moves": len(path),
        "error": str(error) if error else None,
    })

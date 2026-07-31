"""Bounded action-cost search for a short level-9 relay suffix."""

import heapq
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from legs import _bridge_carrier_state


MAX_COST = int(os.environ.get("SUFFIX_MAX_COST", "28"))
MAX_STATES = int(os.environ.get("SUFFIX_MAX_STATES", "3000"))
ENTRY_VARIANT = int(os.environ.get("ENTRY_VARIANT", "0"))


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
    bridges = positions(
        (counts[9] >= 12) & ((counts[9] + counts[1]) == 16)
    )
    pegs = positions(counts[14] >= 12)
    carriers = positions(counts[12] == 16)
    supports = frozenset(
        (row + 1, col)
        for row, col in positions(counts[15] >= 12)
        if row < 60
    )
    state = _bridge_carrier_state(frame)
    holes |= state[0]
    pegs |= state[1]
    carriers |= state[2]
    supports |= state[3]
    return holes, bridges, pegs, carriers, supports


def visible_moves(frame):
    holes, bridges, pegs, carriers, supports = lattice(frame)
    destinations = holes | carriers
    occupied = bridges | pegs | supports
    moves = []
    for kind, pieces in (("bridge", bridges), ("peg", pegs)):
        for source in sorted(pieces):
            for delta_row, delta_col in (
                (-6, 0), (6, 0), (0, -6), (0, 6)
            ):
                midpoint = (
                    source[0] + delta_row,
                    source[1] + delta_col,
                )
                destination = (
                    source[0] + 2 * delta_row,
                    source[1] + 2 * delta_col,
                )
                if midpoint in occupied and destination in destinations:
                    moves.append((kind, source, destination))
    return tuple(moves)


def move_actions(source, destination):
    return (
        [6, source[1] + 1, source[0] + 1],
        [6, destination[1] + 1, destination[0] + 1],
    )


def apply_move(node, move):
    kind, source, destination = move
    child = node.clone()
    for action in move_actions(source, destination):
        play(child, action)
    if child.levels_completed > 8:
        return child
    color = 9 if kind == "bridge" else 14
    array = np.asarray(child.frame())

    def piece_count(position):
        row, col = position
        return int(np.count_nonzero(array[row:row + 4, col:col + 4] == color))

    if piece_count(source) >= 12 or piece_count(destination) < 12:
        return None
    return child


def dense_summary(node):
    _, bridges, pegs, carriers, supports = lattice(node.frame())
    distances = [
        abs(left[0] - right[0]) + abs(left[1] - right[1])
        for left in pegs
        for right in pegs
        if left != right
    ]
    return (
        len(pegs),
        min(distances, default=999),
        len(bridges),
        len(carriers),
        len(supports),
    )


def search(root):
    serial = 0
    queue = [(0, serial, [], root.clone(), None)]
    best_cost = {physical_key(root): 0}
    best_dense = (dense_summary(root), 0, [])
    expanded = 0
    while queue and expanded < MAX_STATES:
        cost, _, path, node, forbidden = heapq.heappop(queue)
        if cost > MAX_COST:
            break
        key = physical_key(node)
        if cost != best_cost.get(key):
            continue
        expanded += 1
        dense = dense_summary(node)
        if dense < best_dense[0]:
            best_dense = (dense, cost, path)
            print("DENSE", {
                "expanded": expanded,
                "cost": cost,
                "dense": dense,
                "tail": path[-8:],
            }, flush=True)
        if node.levels_completed > 8:
            return path, expanded, len(best_cost), best_dense

        children = []
        if cost + 1 <= MAX_COST:
            for action in (3, 4):
                if forbidden == ("key", action):
                    continue
                edge_actions = [action]
                child = node.clone()
                play(child, action)
                if physical_key(child) != key:
                    inverse = ("key", 4 if action == 3 else 3)
                    children.append((1, child, edge_actions, inverse))
        if cost + 2 <= MAX_COST:
            for move in visible_moves(node.frame()):
                if forbidden == ("move", move[0], move[1], move[2]):
                    continue
                edge_actions = list(move_actions(move[1], move[2]))
                child = node.clone()
                for action in edge_actions:
                    play(child, action)
                if child.levels_completed <= 8:
                    color = 9 if move[0] == "bridge" else 14
                    array = np.asarray(child.frame())

                    def piece_count(position):
                        row, col = position
                        return int(np.count_nonzero(
                            array[row:row + 4, col:col + 4] == color
                        ))

                    if (
                        piece_count(move[1]) >= 12
                        or piece_count(move[2]) < 12
                    ):
                        child = None
                if child is not None:
                    inverse = (
                        "move", move[0], move[2], move[1]
                    )
                    children.append((2, child, edge_actions, inverse))

        for edge_cost, child, edge_actions, inverse in children:
            child_cost = cost + edge_cost
            if child_cost > MAX_COST:
                continue
            child_key = physical_key(child)
            if child_cost >= best_cost.get(child_key, MAX_COST + 1):
                continue
            best_cost[child_key] = child_cost
            serial += 1
            child_path = path + edge_actions
            if child.levels_completed > 8:
                return child_path, expanded, len(best_cost), best_dense
            heapq.heappush(
                queue,
                (child_cost, serial, child_path, child, inverse),
            )

        if expanded % 250 == 0:
            print("PROGRESS", {
                "expanded": expanded,
                "seen": len(best_cost),
                "frontier_cost": cost,
            }, flush=True)
    return None, expanded, len(best_cost), best_dense


def probe(env):
    with open("checkpoint.json") as checkpoint_file:
        prefix = json.load(checkpoint_file)["final_path"]
    entry_file = (
        "level9_entry_variant_candidate.json"
        if ENTRY_VARIANT else
        "level9_candidate_102.json"
    )
    with open(entry_file) as candidate_file:
        entry_actions = json.load(candidate_file)[:28]
    for action in prefix:
        play(env, action)
    node = env.clone()
    for action in entry_actions:
        play(node, action)
    print("ENTRY", {
        "variant": ENTRY_VARIANT,
        "dense": dense_summary(node),
        "moves": visible_moves(node.frame()),
    }, flush=True)
    path, expanded, seen, best_dense = search(node)
    print("SEARCH", {
        "path": path,
        "cost": len(path) if path else None,
        "expanded": expanded,
        "seen": seen,
        "best_dense": best_dense,
    }, flush=True)


if __name__ == "__main__":
    levels, path, error = arena.run_program("lf52", probe)
    print("HARNESS", {
        "levels": levels,
        "moves": len(path),
        "error": str(error) if error else None,
    })

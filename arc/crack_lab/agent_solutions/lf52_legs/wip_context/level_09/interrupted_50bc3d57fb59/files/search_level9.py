"""Bounded macro search over reproduced level-9 public observations."""

from collections import Counter
from heapq import heappop, heappush
import json
import sys

import numpy as np

sys.path.insert(0, "/private/tmp/gkm-submitted-protocol-lf52-20260805/arc/crack_lab")

import gkm_arena as arena

from perception import arr, connected_components, safe_step


def key(env):
    return int(env.levels_completed), arr(env.frame())[1:, :].tobytes()


def pegs(env):
    return tuple(
        blob.top_left
        for blob in connected_components(env.frame(), colors=(14,))
        if blob.size == (4, 4)
    )


def carriers(env):
    return tuple(
        blob.top_left
        for blob in connected_components(env.frame(), colors=(12,))
        if blob.size == (4, 4)
    )


def bridges(env):
    frame = arr(env.frame())
    found = []
    for row in range(0, 61, 6):
        for col in range(0, 61, 6):
            block = frame[row:row + 4, col:col + 4]
            if np.count_nonzero(block == 9) == 12 and np.count_nonzero(block == 1) == 4:
                found.append((row, col))
    return tuple(found)


def slots(env):
    return tuple(
        blob.top_left
        for blob in connected_components(env.frame(), colors=(1,))
        if blob.size == (4, 4) and blob.area == 16
    )


def selected(env):
    frame = arr(env.frame())[1:, :]
    return bool(np.any(frame == 2) or np.any(frame == 3))


def piece_move_successors(node):
    root_key = key(node)
    node_pegs = frozenset(pegs(node))
    node_bridges = frozenset(bridges(node))
    destinations = frozenset(slots(node)) | frozenset(carriers(node))
    pieces = tuple(("peg", position) for position in sorted(node_pegs))
    pieces += tuple(("bridge", position) for position in sorted(node_bridges))
    for kind, (row, col) in pieces:
        for dr, dc in ((-6, 0), (6, 0), (0, -6), (0, 6)):
            midpoint = (row + dr, col + dc)
            destination_position = (row + 2 * dr, col + 2 * dc)
            if destination_position not in destinations:
                continue
            if midpoint not in node_pegs | node_bridges:
                continue
            source = (6, col + 1, row + 1)
            destination = (
                6, destination_position[1] + 1, destination_position[0] + 1,
            )
            child = node.clone()
            safe_step(child, source)
            safe_step(child, destination)
            if key(child) != root_key and not selected(child):
                yield child, (source, destination)


def solve_probe(env):
    with open("checkpoint.json") as handle:
        checkpoint = json.load(handle)
    for action in checkpoint["final_path"]:
        env.step(action)

    base_level = int(env.levels_completed)
    root = env.clone()
    serial = 0
    queue = [(0, serial, root, ())]
    best = {key(root): 0}
    counts = Counter()
    summaries = set()
    solution = None
    while queue and len(best) <= 20000:
        cost, _, node, path = heappop(queue)
        if cost != best.get(key(node)):
            continue
        counts[cost] += 1
        summaries.add((pegs(node), bridges(node), carriers(node)))
        if node.levels_completed > base_level:
            solution = path
            break
        if cost >= 56:
            continue

        successors = []
        for action in (1, 2, 3, 4):
            child = node.clone()
            safe_step(child, action)
            if key(child) != key(node):
                successors.append((child, (action,)))
        successors.extend(piece_move_successors(node))
        for child, macro in successors:
            child_cost = cost + len(macro)
            if child_cost > 56:
                continue
            child_key = key(child)
            if child_cost < best.get(child_key, 10 ** 9):
                best[child_key] = child_cost
                serial += 1
                heappush(queue, (child_cost, serial, child, path + macro))

    print("SEARCH", "states", len(best), "expanded", sum(counts.values()))
    print("DEPTH_COUNTS", tuple(sorted(counts.items())))
    print("SUMMARIES", tuple(sorted(summaries)))
    print("SOLUTION", solution)


levels, path, error = arena.run_program("lf52", solve_probe)
print("PROBE_RESULT", levels, len(path), error)

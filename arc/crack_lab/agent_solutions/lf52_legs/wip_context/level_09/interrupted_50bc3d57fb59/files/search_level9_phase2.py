"""Bounded verified macro search for the revealed second phase of level 9."""

from collections import Counter
from heapq import heappop, heappush
import json
import sys

import numpy as np

sys.path.insert(0, "/private/tmp/gkm-submitted-protocol-lf52-20260805/arc/crack_lab")

import gkm_arena as arena

from perception import arr, connected_components, safe_step


PHASE_ONE = (
    (6, 19, 43), (6, 31, 43), (6, 25, 49), (6, 25, 37),
    (6, 25, 43), (6, 25, 31), (6, 25, 37), (6, 25, 25),
    (6, 25, 31), (6, 25, 19), (6, 25, 19), (6, 37, 19),
    (6, 37, 19), (6, 37, 31), (6, 37, 25), (6, 37, 37),
    (6, 37, 31), (6, 37, 43), (6, 37, 37), (6, 37, 49),
    (6, 43, 49), (6, 31, 49), (6, 31, 49), (6, 31, 37),
    (6, 37, 49), (6, 37, 37), (6, 31, 37), (6, 43, 37),
)


def key(env):
    return int(env.levels_completed), arr(env.frame())[1:, :].tobytes()


def selected(env):
    frame = arr(env.frame())[1:, :]
    return bool(np.any(frame == 2) or np.any(frame == 3))


def puzzle_state(frame):
    blobs = connected_components(frame, colors=(1, 9, 11, 14, 15))
    holes = frozenset(
        blob.top_left for blob in blobs
        if blob.color == 1 and blob.size == (4, 4) and blob.area == 16
    )
    pegs = frozenset(
        blob.top_left for blob in blobs
        if blob.color == 14 and blob.size == (4, 4)
    )
    bridges = frozenset(
        blob.top_left for blob in blobs
        if blob.color == 9 and blob.size == (4, 4) and blob.area == 12
    )
    persistent = frozenset(
        (blob.bbox[0] + 1, blob.bbox[1]) for blob in blobs
        if blob.color == 15 and blob.size == (4, 4)
    )
    carriers = frozenset(
        (blob.bbox[0] + 1, blob.bbox[1] + 1) for blob in blobs
        if blob.color == 11 and blob.area >= 4
    )
    return holes, pegs, bridges, persistent, carriers


def piece_successors(node):
    holes, pegs, bridges, persistent, carriers = puzzle_state(node.frame())
    occupied = pegs | bridges
    destinations = holes | (carriers - occupied)
    pieces = tuple(("peg", position) for position in sorted(pegs))
    pieces += tuple(("bridge", position) for position in sorted(bridges))
    for kind, source in pieces:
        for dr, dc in ((-6, 0), (6, 0), (0, -6), (0, 6)):
            midpoint = source[0] + dr, source[1] + dc
            destination_position = source[0] + 2 * dr, source[1] + 2 * dc
            if destination_position not in destinations:
                continue
            if kind == "peg":
                if midpoint not in occupied | persistent:
                    continue
            elif midpoint not in occupied | persistent:
                continue
            source_action = (6, source[1] + 1, source[0] + 1)
            destination_action = (
                6, destination_position[1] + 1, destination_position[0] + 1,
            )
            if not all(0 <= value < 64 for value in source_action[1:] + destination_action[1:]):
                continue
            child = node.clone()
            safe_step(child, source_action)
            safe_step(child, destination_action)
            if key(child) != key(node) and not selected(child):
                yield child, (source_action, destination_action)


def dense_distance(node):
    _, pegs, _, _, _ = puzzle_state(node.frame())
    if len(pegs) < 2:
        return 0
    ordered = tuple(pegs)
    distances = []
    for index, first in enumerate(ordered):
        for second in ordered[index + 1:]:
            distance = abs(first[0] - second[0]) + abs(first[1] - second[1])
            alignment = 0 if first[0] == second[0] or first[1] == second[1] else 2
            distances.append(alignment + max(0, distance // 6 - 1))
    return min(distances)


def probe(env):
    with open("checkpoint.json") as handle:
        checkpoint = json.load(handle)
    for action in checkpoint["final_path"]:
        env.step(action)
    for action in PHASE_ONE:
        safe_step(env, action)

    base_level = int(env.levels_completed)
    root = env.clone()
    queue = [(dense_distance(root), 0, 0, root, ())]
    best = {key(root): 0}
    counts = Counter()
    serial = 0
    solution = None
    while queue and len(best) <= 12000:
        _, cost, _, node, path = heappop(queue)
        if cost != best.get(key(node)):
            continue
        counts[cost] += 1
        if node.levels_completed > base_level:
            solution = path
            break
        if cost >= 28:
            continue

        successors = []
        for action in (1, 2, 3, 4):
            child = node.clone()
            safe_step(child, action)
            if key(child) != key(node):
                successors.append((child, (action,)))
        successors.extend(piece_successors(node))
        for child, macro in successors:
            child_cost = cost + len(macro)
            if child_cost > 28:
                continue
            child_key = key(child)
            if child_cost < best.get(child_key, 10 ** 9):
                best[child_key] = child_cost
                serial += 1
                priority = child_cost + dense_distance(child)
                heappush(queue, (priority, child_cost, serial, child, path + macro))

    print("PHASE2_SEARCH", "states", len(best), "expanded", sum(counts.values()))
    print("PHASE2_DEPTHS", tuple(sorted(counts.items())))
    print("PHASE2_SOLUTION", solution)
    if solution is not None:
        clone = root.clone()
        for action in solution:
            safe_step(clone, action)
        print("PHASE2_REPLAY", clone.levels_completed, len(solution))


levels, path, error = arena.run_program("lf52", probe)
print("PROBE_RESULT", levels, len(path), error)

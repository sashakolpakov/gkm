"""Bounded action-cost search over level 7's original final board."""

from heapq import heappop, heappush
import json
import os

import gkm_try

from legs import _movable_bridge_board
from perception import arr, connected_components, safe_step


LEVEL_START = 331
LEVEL_END = 476
START_GROUP = 17
MAX_STATES = int(os.environ.get("MAX_STATES", "4000"))


def groups(segment):
    result = []; index = 0
    while index < len(segment):
        keys = []
        while index < len(segment) and not isinstance(segment[index], list):
            keys.append(segment[index]); index += 1
        pair = []
        while index < len(segment) and isinstance(segment[index], list) and len(pair) < 2:
            pair.append(tuple(segment[index])); index += 1
        result.append((tuple(keys), tuple(pair)))
    return result


def frame_key(node):
    return arr(node.frame())[1:, :].tobytes()


def legal_moves(node):
    slots, carriers, bridges, pegs = _movable_bridge_board(node.frame())
    fixed = {
        (blob.bbox[0] + 1, blob.bbox[1])
        for blob in connected_components(node.frame(), colors=(15,))
        if blob.size == (4, 4) and blob.area >= 12
    }
    occupied = bridges | pegs
    result = []
    for source in sorted(occupied):
        for dr, dc in ((-12, 0), (12, 0), (0, -12), (0, 12)):
            destination = source[0] + dr, source[1] + dc
            midpoint = source[0] + dr // 2, source[1] + dc // 2
            if (
                destination in slots | carriers
                and destination not in occupied
                and midpoint in occupied | fixed
            ):
                result.append((source, destination))
    return tuple(result)


def lower_bound(node):
    pegs = _movable_bridge_board(node.frame())[3]
    return 2 * max(0, len(pegs) - 1)


def reconstruct(parent, cursor):
    macros = []
    while parent[cursor] is not None:
        cursor, macro = parent[cursor]; macros.append(macro)
    macros.reverse()
    return [action for macro in macros for action in macro]


def probe(env):
    with open("checkpoint.json") as checkpoint_file:
        full = json.load(checkpoint_file)["final_path"]
    for action in full[:LEVEL_START]:
        safe_step(env, action)
    level_groups = groups(full[LEVEL_START:LEVEL_END])
    for keys, pair in level_groups[:START_GROUP]:
        for action in keys + pair:
            safe_step(env, action)
    known_path = [action for keys, pair in level_groups[START_GROUP:] for action in keys + pair]
    base_level = int(env.levels_completed)

    known = {}; trace = env.clone()
    known[frame_key(trace)] = tuple(known_path)
    for index, action in enumerate(known_path):
        safe_step(trace, action)
        known.setdefault(frame_key(trace), tuple(known_path[index + 1:]))

    root = env.clone(); root_key = frame_key(root)
    distance = {root_key: 0}; parent = {root_key: None}; nodes = {root_key: root}
    serial = 0; queue = [(lower_bound(root), 0, serial, root_key)]
    upper = len(known_path); best_path = known_path; popped = 0

    while queue and popped < MAX_STATES:
        _, cost, _, state_key = heappop(queue)
        if cost != distance.get(state_key):
            continue
        node = nodes.pop(state_key); popped += 1
        suffix = known.get(state_key)
        if suffix is not None and cost + len(suffix) < upper:
            upper = cost + len(suffix)
            best_path = reconstruct(parent, state_key) + list(suffix)
            print("L7_FINAL_BOUND", popped, len(distance), upper, flush=True)
        if popped % 250 == 0:
            print("L7_FINAL_PROGRESS", popped, len(distance), cost, upper, flush=True)
        if cost + lower_bound(node) >= upper:
            continue

        macros = [((action,), 1) for action in (1, 2, 3, 4)]
        macros += [(
            (
                (6, source[1] + 1, source[0] + 1),
                (6, destination[1] + 1, destination[0] + 1),
            ),
            2,
        ) for source, destination in legal_moves(node)]
        for macro, weight in macros:
            child = node.clone()
            for action in macro:
                safe_step(child, action)
            child_cost = cost + weight; child_key = frame_key(child)
            if child_key == state_key or child_cost >= distance.get(child_key, upper):
                continue
            distance[child_key] = child_cost; parent[child_key] = state_key, macro
            if int(child.levels_completed) > base_level:
                if child_cost < upper:
                    upper = child_cost; best_path = reconstruct(parent, child_key)
                    print("L7_FINAL_GOAL", popped, len(distance), upper, flush=True)
                continue
            nodes[child_key] = child; serial += 1
            priority = child_cost + lower_bound(child)
            heappush(queue, (priority, child_cost, serial, child_key))

    replay = env.clone()
    for action in best_path:
        safe_step(replay, action)
    print(
        "L7_FINAL_DIJKSTRA", popped, len(distance), len(best_path),
        int(replay.levels_completed), bool(queue), best_path,
    )


gkm_try.A.run_program("lf52", probe)

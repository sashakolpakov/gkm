"""Find the globally shortest carrier schedule for a verified peg route."""

import heapq
import json
import os

import gkm_try
from legs import _bridge_carrier_moves, _bridge_carrier_state
from perception import arr


TARGET_LEVEL = int(os.environ.get("TARGET_LEVEL", "5"))
MAX_STATES = int(os.environ.get("MAX_STATES", "5000"))
INCUMBENT = int(os.environ.get("INCUMBENT", "89"))
LEVEL_ENDS = {1: 8, 2: 42, 3: 87, 4: 149, 5: 238, 6: 331, 7: 476, 8: 544}


def frame_key(node):
    return _bridge_carrier_state(node.frame())


def coordinate_pairs(segment):
    pairs = []; index = 0
    while index < len(segment):
        while index < len(segment) and not isinstance(segment[index], list): index += 1
        pair = []
        while index < len(segment) and isinstance(segment[index], list) and len(pair) < 2:
            pair.append(tuple(segment[index])); index += 1
        if len(pair) == 2: pairs.append(tuple(pair))
    return tuple(pairs)


def desired(pair):
    return (
        (pair[0][2] - 1, pair[0][1] - 1),
        (pair[1][2] - 1, pair[1][1] - 1),
    )


def probe(env):
    with open("checkpoint.json") as checkpoint_file:
        path = json.load(checkpoint_file)["final_path"]
    start = LEVEL_ENDS[TARGET_LEVEL - 1]; end = LEVEL_ENDS[TARGET_LEVEL]
    for action in path[:start]: env.step(action)
    pairs = coordinate_pairs(path[start:end]); base_level = env.levels_completed
    root_key = (0, frame_key(env)); distance = {root_key: 0}; serial = 0
    queue = [(0, serial, env.clone(), 0, ())]; found = None
    while queue and len(distance) <= MAX_STATES:
        cost, _, node, pair_index, action_path = heapq.heappop(queue)
        state_key = pair_index, frame_key(node)
        if cost != distance.get(state_key): continue
        if node.levels_completed > base_level:
            found = action_path; break
        remaining_clicks = 2 * (len(pairs) - pair_index)
        if cost + remaining_clicks >= INCUMBENT: continue
        for action in (1, 2, 3, 4):
            child = node.clone(); child.step(action); child_state = pair_index, frame_key(child)
            child_cost = cost + 1
            if child_state == state_key or child_cost >= distance.get(child_state, 10 ** 9): continue
            distance[child_state] = child_cost; serial += 1
            heapq.heappush(queue, (child_cost, serial, child, pair_index, action_path + (action,)))
        if pair_index >= len(pairs): continue
        move = desired(pairs[pair_index])
        legal = {(source, destination) for _, source, destination in _bridge_carrier_moves(node.frame())}
        if move not in legal: continue
        child = node.clone()
        for action in pairs[pair_index]: child.step(*action)
        child_state = pair_index + 1, frame_key(child); child_cost = cost + 2
        if child_cost < distance.get(child_state, 10 ** 9):
            distance[child_state] = child_cost; serial += 1
            heapq.heappush(
                queue,
                (child_cost, serial, child, pair_index + 1, action_path + pairs[pair_index]),
            )
    print("FIXED_ROUTE", TARGET_LEVEL, len(distance), None if found is None else (len(found), found))


gkm_try.A.run_program("lf52", probe)

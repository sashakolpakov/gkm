"""Bounded exact rejoin search for the alternate middle-region unload."""

import heapq
import json

import gkm_try
from legs import _movable_bridge_board
from perception import arr, connected_components


MAX_COST = 13
MAX_STATES = 1200


def key(node):
    return arr(node.frame())[1:, :].tobytes()


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


def piece_macros(node):
    slots, carriers, bridges, pegs = _movable_bridge_board(node.frame())
    fixed = {
        (blob.bbox[0] + 1, blob.bbox[1])
        for blob in connected_components(node.frame(), colors=(15,))
        if blob.size == (4, 4) and blob.area >= 12
    }
    occupied = bridges | pegs
    for source in sorted(occupied):
        for dr, dc in ((-12, 0), (12, 0), (0, -12), (0, 12)):
            destination = source[0] + dr, source[1] + dc
            midpoint = source[0] + dr // 2, source[1] + dc // 2
            if destination in slots | carriers and destination not in occupied and midpoint in occupied | fixed:
                yield (
                    (6, source[1] + 1, source[0] + 1),
                    (6, destination[1] + 1, destination[0] + 1),
                )


def probe(env):
    with open("checkpoint.json") as checkpoint_file:
        path = json.load(checkpoint_file)["final_path"]
    for action in path[:331]: env.step(action)
    level_groups = groups(path[331:476])
    alternate = env.clone(); target = env.clone()
    for keys, pair in level_groups[:10]:
        for action in keys: alternate.step(action)
        for action in pair: alternate.step(*action)
    for action in (1, 3, 3, 1, 1, 4, 4, 4, 2): alternate.step(action)
    for action in ((6, 29, 43), (6, 29, 55)): alternate.step(*action)
    for keys, pair in level_groups[:13]:
        for action in keys: target.step(action)
        for action in pair: target.step(*action)
    target_key = key(target)

    serial = 0; root_key = key(alternate); distance = {root_key: 0}
    queue = [(0, serial, alternate.clone(), ())]; found = None
    while queue and len(distance) <= MAX_STATES:
        cost, _, node, action_path = heapq.heappop(queue)
        node_key = key(node)
        if cost != distance[node_key]: continue
        if node_key == target_key:
            found = action_path; break
        if cost >= MAX_COST: continue
        macros = tuple((action,) for action in (1, 2, 3, 4)) + tuple(piece_macros(node))
        for macro in macros:
            child = node.clone()
            for action in macro:
                if isinstance(action, tuple): child.step(*action)
                else: child.step(action)
            child_key = key(child); child_cost = cost + len(macro)
            if child_key == node_key or child_cost >= distance.get(child_key, 10 ** 9): continue
            distance[child_key] = child_cost; serial += 1
            heapq.heappush(queue, (child_cost, serial, child, action_path + macro))
    print("ALT_REJOIN", len(distance), found)


gkm_try.A.run_program("lf52", probe)

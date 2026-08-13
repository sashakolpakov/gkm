"""Compact carrier-neighbour terrain observations at level-7 boundaries."""

from collections import Counter
import json

import gkm_try

from legs import _movable_bridge_board
from perception import arr, safe_step


def tile(frame, point):
    row, col = point
    if not (0 <= row <= 58 and 0 <= col <= 58):
        return ()
    return tuple(sorted(Counter(arr(frame)[row:row + 6, col:col + 6].ravel().tolist()).items()))


def carrier_observation(node):
    frame = node.frame(); carriers = _movable_bridge_board(frame)[1]
    rows = []
    for carrier in sorted(carriers):
        row, col = carrier
        neighbours = []
        for action, delta in ((1, (-6, 0)), (2, (6, 0)), (3, (0, -6)), (4, (0, 6))):
            point = row + delta[0], col + delta[1]
            child = node.clone(); safe_step(child, action)
            neighbours.append((action, point, tile(frame, point), tuple(sorted(_movable_bridge_board(child.frame())[1]))))
        rows.append((carrier, tile(frame, carrier), tuple(neighbours)))
    return tuple(rows)


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


def probe(env):
    with open("checkpoint.json") as checkpoint_file:
        path = json.load(checkpoint_file)["final_path"]
    for action in path[:331]:
        safe_step(env, action)
    print("TERRAIN_L7_ENTRY", carrier_observation(env))
    level_groups = groups(path[331:476])
    for keys, pair in level_groups[:17]:
        for action in keys + pair:
            safe_step(env, action)
    print("TERRAIN_L7_FINAL", carrier_observation(env))


gkm_try.A.run_program("lf52", probe)

"""Enumerate movable-piece jumps reachable through key-only carrier states."""

import json
import os
from collections import deque

import gkm_try
from legs import _movable_bridge_board
from perception import arr, connected_components


TARGET_LEVEL = int(os.environ.get("TARGET_LEVEL", "7"))
TARGET_GROUP = int(os.environ.get("TARGET_GROUP", "0"))
MAX_STATES = int(os.environ.get("MAX_STATES", "600"))
MAX_DEPTH = int(os.environ.get("MAX_DEPTH", "18"))
LEVEL_ENDS = {1: 8, 2: 42, 3: 87, 4: 149, 5: 238, 6: 331, 7: 476, 8: 544}


def frame_key(node):
    return arr(node.frame())[1:, :].tobytes()


def moves(node):
    slots, carriers, bridges, pegs = _movable_bridge_board(node.frame())
    if TARGET_LEVEL >= 8:
        bridges |= {
            blob.top_left
            for blob in connected_components(node.frame(), colors=(9,))
            if blob.size == (4, 4) and blob.area == 12
        }
    fixed = {
        (blob.bbox[0] + 1, blob.bbox[1])
        for blob in connected_components(node.frame(), colors=(15,))
        if blob.size == (4, 4) and blob.area >= 12
    }
    occupied = bridges | pegs
    result = []
    for kind, sources in (("B", bridges), ("P", pegs)):
        for source in sorted(sources):
            for dr, dc in ((-12, 0), (12, 0), (0, -12), (0, 12)):
                destination = source[0] + dr, source[1] + dc
                midpoint = source[0] + dr // 2, source[1] + dc // 2
                if destination in slots | carriers and destination not in occupied and midpoint in occupied | fixed:
                    result.append((kind, source, destination))
    return tuple(result)


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
    level_start = LEVEL_ENDS[TARGET_LEVEL - 1]
    level_end = LEVEL_ENDS[TARGET_LEVEL]
    for action in path[:level_start]: env.step(action)
    level_groups = groups(path[level_start:level_end])
    for keys, pair in level_groups[:TARGET_GROUP]:
        for action in keys: env.step(action)
        for action in pair: env.step(*action)
    root = env.clone(); queue = deque([(root.clone(), ())]); seen = {frame_key(root)}; found = {}
    while queue and len(seen) <= MAX_STATES:
        node, key_path = queue.popleft()
        for move in moves(node): found.setdefault(move, key_path)
        if len(key_path) >= MAX_DEPTH: continue
        for action in (1, 2, 3, 4):
            child = node.clone(); child.step(action); child_key = frame_key(child)
            if child_key in seen: continue
            seen.add(child_key); queue.append((child, key_path + (action,)))
    print("MOVABLE_FRONTIER", TARGET_LEVEL, TARGET_GROUP, len(seen), tuple(sorted(found.items())))


gkm_try.A.run_program("lf52", probe)

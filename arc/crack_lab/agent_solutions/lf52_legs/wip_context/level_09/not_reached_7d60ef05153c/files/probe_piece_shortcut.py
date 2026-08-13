"""Exact bounded search for shorter coordinate-only relay stages."""

import json
import os
from collections import deque

import gkm_try
from legs import _movable_bridge_board
from perception import arr, connected_components


TARGET_LEVEL = int(os.environ.get("TARGET_LEVEL", "6"))
START_GROUP = int(os.environ.get("START_GROUP", "11"))
END_GROUP = int(os.environ.get("END_GROUP", "22"))
MAX_STATES = int(os.environ.get("MAX_STATES", "5000"))
LEVEL_ENDS = {1: 8, 2: 42, 3: 87, 4: 149, 5: 238, 6: 331, 7: 476, 8: 544}


def frame_key(node):
    return arr(node.frame())[1:, :].tobytes()


def groups(segment):
    result = []
    index = 0
    while index < len(segment):
        keys = []
        while index < len(segment) and not isinstance(segment[index], list):
            keys.append(segment[index]); index += 1
        pair = []
        while index < len(segment) and isinstance(segment[index], list) and len(pair) < 2:
            pair.append(tuple(segment[index])); index += 1
        result.append((tuple(keys), tuple(pair)))
    return result


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
    for source in sorted(occupied):
        for dr, dc in ((-12, 0), (12, 0), (0, -12), (0, 12)):
            destination = source[0] + dr, source[1] + dc
            midpoint = source[0] + dr // 2, source[1] + dc // 2
            if (
                destination in slots | carriers
                and destination not in occupied
                and midpoint in occupied | fixed
            ):
                yield source, destination


def probe(env):
    with open("checkpoint.json") as checkpoint_file:
        path = json.load(checkpoint_file)["final_path"]
    level_start = LEVEL_ENDS[TARGET_LEVEL - 1]
    level_end = LEVEL_ENDS[TARGET_LEVEL]
    for action in path[:level_start]: env.step(action)
    level_groups = groups(path[level_start:level_end])
    start = env.clone(); target = env.clone()
    for index, (keys, pair) in enumerate(level_groups):
        if index < START_GROUP:
            for action in keys: start.step(action)
            for action in pair: start.step(*action)
        elif index == START_GROUP:
            for action in keys: start.step(action)
        if index < END_GROUP:
            for action in keys: target.step(action)
            for action in pair: target.step(*action)
    target_key = frame_key(target)
    original_moves = END_GROUP - START_GROUP
    queue = deque([(start.clone(), ())]); seen = {frame_key(start)}; found = None
    while queue and len(seen) <= MAX_STATES:
        node, macro_path = queue.popleft()
        if frame_key(node) == target_key and macro_path:
            found = macro_path; break
        if node.levels_completed >= TARGET_LEVEL:
            found = macro_path; break
        if len(macro_path) >= original_moves - 1:
            continue
        for source, destination in moves(node):
            child = node.clone()
            pair = (
                (6, source[1] + 1, source[0] + 1),
                (6, destination[1] + 1, destination[0] + 1),
            )
            for action in pair: child.step(*action)
            child_key = frame_key(child)
            if child_key in seen or child_key == frame_key(node): continue
            seen.add(child_key); queue.append((child, macro_path + (pair,)))
    print("PIECE_RESULT", TARGET_LEVEL, START_GROUP, END_GROUP, original_moves, len(seen), found)


gkm_try.A.run_program("lf52", probe)

"""Optimize carrier alignments while preserving whole verified move blocks."""

import json
import os
from collections import deque

import gkm_try
from legs import _bridge_carrier_moves, _movable_bridge_board
from perception import arr, connected_components


TARGET_LEVEL = int(os.environ.get("TARGET_LEVEL", "6"))
MAX_STATES = int(os.environ.get("MAX_STATES", "1600"))
MAX_DEPTH = int(os.environ.get("MAX_DEPTH", "20"))
START_GROUP = int(os.environ.get("START_GROUP", "0"))
ONE_BLOCK = os.environ.get("ONE_BLOCK") == "1"
CHECK_ONLY = os.environ.get("CHECK_ONLY") == "1"
LEVEL_ENDS = {1: 8, 2: 42, 3: 87, 4: 149, 5: 238, 6: 331, 7: 476, 8: 544}


def is_click(action):
    return isinstance(action, (list, tuple)) and len(action) == 3


def frame_key(node):
    return arr(node.frame())[1:, :].tobytes()


def desired_is_legal(node, move):
    frame = node.frame()
    if TARGET_LEVEL <= 5:
        return move in {
            (source, destination)
            for _, source, destination in _bridge_carrier_moves(frame)
        }
    source, destination = move
    slots, carriers, bridges, pegs = _movable_bridge_board(frame)
    if TARGET_LEVEL >= 8:
        bridges |= {
            blob.top_left
            for blob in connected_components(frame, colors=(9,))
            if blob.size == (4, 4) and blob.area == 12
        }
    fixed = {
        (blob.bbox[0] + 1, blob.bbox[1])
        for blob in connected_components(frame, colors=(15,))
        if blob.size == (4, 4) and blob.area >= 12
    }
    midpoint = (
        (source[0] + destination[0]) // 2,
        (source[1] + destination[1]) // 2,
    )
    return (
        source in pegs | bridges
        and destination in slots | carriers
        and destination not in pegs | bridges
        and midpoint in pegs | bridges | fixed
    )


def desired(pair):
    return (
        (pair[0][2] - 1, pair[0][1] - 1),
        (pair[1][2] - 1, pair[1][1] - 1),
    )


def execute_block(node, pairs):
    trial = node.clone()
    for pair in pairs:
        if not desired_is_legal(trial, desired(pair)):
            return None
        for action in pair:
            trial.step(*action)
    return trial


def shortest_block(root, pairs):
    queue = deque([(root.clone(), ())])
    seen = {frame_key(root)}
    while queue and len(seen) <= MAX_STATES:
        node, path = queue.popleft()
        result = execute_block(node, pairs)
        if result is not None:
            return path, result, len(seen)
        if len(path) >= MAX_DEPTH:
            continue
        for action in (1, 2, 3, 4):
            child = node.clone(); child.step(action)
            child_key = frame_key(child)
            if child_key in seen:
                continue
            seen.add(child_key)
            queue.append((child, path + (action,)))
    return None, None, len(seen)


def probe(env):
    with open("checkpoint.json") as checkpoint_file:
        full_path = json.load(checkpoint_file)["final_path"]
    start = LEVEL_ENDS[TARGET_LEVEL - 1]
    end = LEVEL_ENDS[TARGET_LEVEL]
    for action in full_path[:start]: env.step(action)
    entry = env.clone()
    segment = full_path[start:end]

    groups = []
    index = 0
    while index < len(segment):
        keys = []
        while index < len(segment) and not is_click(segment[index]):
            keys.append(segment[index]); index += 1
        pair = []
        while index < len(segment) and is_click(segment[index]) and len(pair) < 2:
            pair.append(tuple(segment[index])); index += 1
        if len(pair) == 2:
            groups.append((tuple(keys), tuple(pair)))

    node = entry.clone()
    solution = []
    for keys, pair in groups[:START_GROUP]:
        for action in keys: node.step(action)
        for action in pair: node.step(*action)
        solution.extend(keys); solution.extend(pair)
    trace = []
    first = START_GROUP
    while first < len(groups):
        last = first + 1
        while last < len(groups) and not groups[last][0]:
            last += 1
        pairs = tuple(pair for _, pair in groups[first:last])
        print("BLOCK_START", first, last, len(groups[first][0]), flush=True)
        if CHECK_ONLY:
            alignment, result, searched = None, None, 0
        else:
            alignment, result, searched = shortest_block(node, pairs)
        if alignment is None:
            check = node.clone()
            for action in groups[first][0]: check.step(action)
            checks = []
            for pair in pairs:
                move = desired(pair)
                checks.append((move, desired_is_legal(check, move)))
                for action in pair: check.step(*action)
            print("BLOCK_ORIGINAL_CHECK", checks, flush=True)
        print(
            "BLOCK_DONE", first, last,
            None if alignment is None else len(alignment), searched,
            flush=True,
        )
        trace.append((first, last, len(groups[first][0]), None if alignment is None else len(alignment), searched))
        if alignment is None:
            break
        solution.extend(alignment)
        for pair in pairs: solution.extend(pair)
        node = result
        if node.levels_completed >= TARGET_LEVEL:
            break
        first = last
        if ONE_BLOCK:
            break

    validation = entry.clone()
    for action in solution:
        if isinstance(action, tuple): validation.step(*action)
        else: validation.step(action)
        if validation.levels_completed >= TARGET_LEVEL: break
    valid = validation.levels_completed >= TARGET_LEVEL
    print("BLOCK_RESULT", TARGET_LEVEL, len(segment), len(solution), valid, trace)
    if valid: print("BLOCK_PATH", solution)


gkm_try.A.run_program("lf52", probe)

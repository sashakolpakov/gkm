"""Replay verified peg subgoals using shortest reachable carrier alignments."""

import json
import os
from collections import deque

import gkm_try
from legs import _bridge_carrier_moves, _movable_bridge_board
from perception import arr, connected_components


TARGET_LEVEL = int(os.environ.get("TARGET_LEVEL", "5"))
MAX_STATES = int(os.environ.get("MAX_STATES", "1000"))
MAX_DEPTH = int(os.environ.get("MAX_DEPTH", "18"))
LEVEL_ENDS = {1: 8, 2: 42, 3: 87, 4: 149, 5: 238, 6: 331, 7: 476, 8: 544}


def is_click(action):
    return isinstance(action, (list, tuple)) and len(action) == 3


def frame_key(node):
    return arr(node.frame())[1:, :].tobytes()


def desired_is_legal(node, desired):
    frame = node.frame()
    if TARGET_LEVEL <= 5:
        return desired in {
            (source, destination)
            for _, source, destination in _bridge_carrier_moves(frame)
        }
    source, destination = desired
    slots, carriers, bridges, pegs = _movable_bridge_board(frame)
    if TARGET_LEVEL >= 8:
        bridges |= {
            blob.top_left
            for blob in connected_components(frame, colors=(9,))
            if blob.size == (4, 4) and blob.area >= 12
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
    plausible = (
        source in pegs | bridges
        and destination not in pegs | bridges
        and midpoint in pegs | bridges | fixed
    )
    if not plausible:
        return False
    trial = node.clone()
    before = frame_key(trial)
    for action in (
        (6, source[1] + 1, source[0] + 1),
        (6, destination[1] + 1, destination[0] + 1),
    ):
        trial.step(*action)
    selected = any(
        blob.bbox[0] >= 1 and blob.area >= 4
        for blob in connected_components(trial.frame(), colors=(3,))
    )
    return not selected and frame_key(trial) != before


def shortest_alignment(root, desired):
    queue = deque([(root.clone(), ())])
    seen = {frame_key(root)}
    while queue and len(seen) <= MAX_STATES:
        node, path = queue.popleft()
        if desired_is_legal(node, desired):
            return path, len(seen)
        if len(path) >= MAX_DEPTH:
            continue
        for action in (1, 2, 3, 4):
            child = node.clone()
            child.step(action)
            child_key = frame_key(child)
            if child_key in seen:
                continue
            seen.add(child_key)
            queue.append((child, path + (action,)))
    return None, len(seen)


def probe(env):
    with open("checkpoint.json") as checkpoint_file:
        full_path = json.load(checkpoint_file)["final_path"]
    start = LEVEL_ENDS[TARGET_LEVEL - 1]
    end = LEVEL_ENDS[TARGET_LEVEL]
    for action in full_path[:start]:
        env.step(action)
    entry = env.clone()
    segment = full_path[start:end]
    pairs = []
    index = 0
    while index < len(segment):
        while index < len(segment) and not is_click(segment[index]):
            index += 1
        clicks = []
        while index < len(segment) and is_click(segment[index]) and len(clicks) < 2:
            clicks.append(tuple(segment[index])); index += 1
        if len(clicks) == 2:
            pairs.append(tuple(clicks))

    node = entry.clone()
    solution = []
    trace = []
    for pair_index, pair in enumerate(pairs):
        desired = (
            (pair[0][2] - 1, pair[0][1] - 1),
            (pair[1][2] - 1, pair[1][1] - 1),
        )
        alignment, searched = shortest_alignment(node, desired)
        if alignment is None:
            trace.append((pair_index, desired, None, searched))
            break
        for action in alignment:
            node.step(action); solution.append(action)
        for action in pair:
            node.step(*action); solution.append(action)
        trace.append((pair_index, desired, len(alignment), searched))
        if node.levels_completed >= TARGET_LEVEL:
            break

    validation = entry.clone()
    for action in solution:
        if isinstance(action, tuple): validation.step(*action)
        else: validation.step(action)
        if validation.levels_completed >= TARGET_LEVEL: break
    print(
        "RETARGET_RESULT", TARGET_LEVEL, len(segment), len(solution),
        validation.levels_completed >= TARGET_LEVEL, trace,
    )
    if validation.levels_completed >= TARGET_LEVEL:
        print("RETARGET_PATH", solution)


gkm_try.A.run_program("lf52", probe)

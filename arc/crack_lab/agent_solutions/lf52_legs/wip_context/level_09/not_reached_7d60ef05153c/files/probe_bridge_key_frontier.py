"""Enumerate distinct peg moves reachable by carrier keys before a macro."""

import json
import os
from collections import deque

import gkm_try
from legs import _bridge_carrier_moves, _bridge_carrier_state
from perception import arr


TARGET_LEVEL = int(os.environ.get("TARGET_LEVEL", "5"))
TARGET_GROUP = int(os.environ.get("TARGET_GROUP", "0"))
MAX_STATES = int(os.environ.get("MAX_STATES", "1000"))
MAX_DEPTH = int(os.environ.get("MAX_DEPTH", "18"))
LEVEL_ENDS = {1: 8, 2: 42, 3: 87, 4: 149, 5: 238, 6: 331, 7: 476, 8: 544}


def frame_key(node):
    return arr(node.frame())[1:, :].tobytes()


def probe(env):
    with open("checkpoint.json") as checkpoint_file:
        path = json.load(checkpoint_file)["final_path"]
    start = LEVEL_ENDS[TARGET_LEVEL - 1]
    end = LEVEL_ENDS[TARGET_LEVEL]
    for action in path[:start]: env.step(action)
    segment = path[start:end]; index = group = 0
    while index < len(segment) and group < TARGET_GROUP:
        while index < len(segment) and not isinstance(segment[index], list):
            env.step(segment[index]); index += 1
        count = 0
        while index < len(segment) and isinstance(segment[index], list) and count < 2:
            env.step(*segment[index]); index += 1; count += 1
        group += 1
    root = env.clone(); queue = deque([(root.clone(), ())]); seen = {frame_key(root)}
    found = {}; candidate_count = 0
    while queue and len(seen) <= MAX_STATES:
        node, key_path = queue.popleft()
        for move in _bridge_carrier_moves(node.frame()):
            candidate_count += 1
            signature = move, _bridge_carrier_state(node.frame())[2:5]
            found.setdefault(signature, key_path)
        if len(key_path) >= MAX_DEPTH: continue
        for action in (1, 2, 3, 4):
            child = node.clone(); child.step(action); child_key = frame_key(child)
            if child_key in seen: continue
            seen.add(child_key); queue.append((child, key_path + (action,)))
    summary = {}
    for (move, geometry), key_path in found.items():
        current = summary.get(move)
        item = (len(key_path), key_path, geometry)
        if current is None or item[0] < current[0]: summary[move] = item
    print("BRIDGE_FRONTIER", TARGET_LEVEL, TARGET_GROUP, len(seen), candidate_count, tuple(sorted(summary.items())))


gkm_try.A.run_program("lf52", probe)

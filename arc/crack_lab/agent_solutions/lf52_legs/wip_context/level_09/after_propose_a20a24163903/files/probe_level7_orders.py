"""Test cooperative load/unload orders for the level-seven carrier pair."""

import json
import os
from collections import deque

import gkm_try
from legs import _movable_bridge_board
from perception import arr, connected_components


MAX_STATES = int(os.environ.get("MAX_STATES", "400"))
MAX_DEPTH = int(os.environ.get("MAX_DEPTH", "16"))
MODE = os.environ.get("MODE", "A")
LEVEL_START = 331
LEVEL_END = 476

MOVES = {
    "PL": ((6, 7, 13), (6, 7, 25)),
    "PU": ((6, 13, 43), (6, 13, 55)),
    "BL": ((6, 43, 13), (6, 43, 25)),
    "BU": ((6, 43, 43), (6, 43, 55)),
}
ORDERS = (
    ("PL", "BL", "PU", "BU"),
    ("PL", "BL", "BU", "PU"),
    ("BL", "PL", "PU", "BU"),
    ("BL", "PL", "BU", "PU"),
    ("BL", "BU", "PL", "PU"),
)


def state_key(node):
    return arr(node.frame())[1:, :].tobytes()


def desired(pair):
    return (
        (pair[0][2] - 1, pair[0][1] - 1),
        (pair[1][2] - 1, pair[1][1] - 1),
    )


def legal(node, move):
    source, destination = move
    slots, carriers, bridges, pegs = _movable_bridge_board(node.frame())
    fixed = {
        (blob.bbox[0] + 1, blob.bbox[1])
        for blob in connected_components(node.frame(), colors=(15,))
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


def legal_moves(node):
    slots, carriers, bridges, pegs = _movable_bridge_board(node.frame())
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


def align(root, pair):
    move = desired(pair)
    queue = deque([(root.clone(), ())])
    seen = {state_key(root)}
    while queue and len(seen) <= MAX_STATES:
        node, path = queue.popleft()
        if legal(node, move):
            return path, len(seen)
        if len(path) >= MAX_DEPTH:
            continue
        for action in (1, 2, 3, 4):
            child = node.clone(); child.step(action)
            child_key = state_key(child)
            if child_key in seen:
                continue
            seen.add(child_key)
            queue.append((child, path + (action,)))
    return None, len(seen)


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


def probe(env):
    with open("checkpoint.json") as checkpoint_file:
        path = json.load(checkpoint_file)["final_path"]
    for action in path[:LEVEL_START]: env.step(action)
    entry = env.clone()
    original_groups = groups(path[LEVEL_START:LEVEL_END])
    if MODE in ("D", "E"):
        node = entry.clone()
        for keys, pair in original_groups[:10]:
            for action in keys: node.step(action)
            for action in pair: node.step(*action)
        alternate_keys = (1, 3, 3, 1, 1, 4, 4, 4, 2)
        for action in alternate_keys: node.step(action)
        alternate_pair = ((6, 29, 43), (6, 29, 55))
        for action in alternate_pair: node.step(*action)
        if MODE == "E":
            targets = (
                ((54, 28), (54, 40)),
                ((54, 16), (42, 16)),
                ((54, 16), (54, 4)),
            )
            queue = deque([(node.clone(), ())]); seen = {state_key(node)}; found = {}
            while queue and len(seen) <= MAX_STATES and len(found) < len(targets):
                current, key_path = queue.popleft()
                for target in targets:
                    if target not in found and legal(current, target): found[target] = key_path
                if len(key_path) >= MAX_DEPTH: continue
                for action in (1, 2, 3, 4):
                    child = current.clone(); child.step(action); child_key = state_key(child)
                    if child_key in seen: continue
                    seen.add(child_key); queue.append((child, key_path + (action,)))
            print("ORDER_FRONTIER", len(seen), tuple(sorted(found.items())), flush=True)
            return
        queue = deque([(node.clone(), ())]); seen = {state_key(node)}; found = {}
        while queue and len(seen) <= MAX_STATES:
            current, key_path = queue.popleft()
            for move in legal_moves(current): found.setdefault(move, key_path)
            if len(key_path) >= MAX_DEPTH: continue
            for action in (1, 2, 3, 4):
                child = current.clone(); child.step(action); child_key = state_key(child)
                if child_key in seen: continue
                seen.add(child_key); queue.append((child, key_path + (action,)))
        print("ORDER_FRONTIER", len(seen), tuple(sorted(found.items())), flush=True)
        return
    if MODE == "C":
        stage_entry = entry.clone()
        for keys, pair in original_groups[:11]:
            for action in keys: stage_entry.step(action)
            for action in pair: stage_entry.step(*action)
        queue = deque([(stage_entry.clone(), ())]); seen = {state_key(stage_entry)}; found = {}
        while queue and len(seen) <= MAX_STATES:
            node, key_path = queue.popleft()
            for move in legal_moves(node):
                if move[0] == "P" and move[2] not in ((54, 16),):
                    found.setdefault(move, key_path)
            if len(key_path) >= MAX_DEPTH: continue
            for action in (1, 2, 3, 4):
                child = node.clone(); child.step(action); child_key = state_key(child)
                if child_key in seen: continue
                seen.add(child_key); queue.append((child, key_path + (action,)))
        print("ORDER_FRONTIER", len(seen), tuple(sorted((move, path) for move, path in found.items())), flush=True)
        return
    if MODE == "B":
        stage_moves = {
            "BU": original_groups[11][1],
            "PL": original_groups[12][1],
            "PU": original_groups[13][1],
        }
        orders = (
            ("PL", "BU", "PU"),
            ("PL", "PU", "BU"),
        )
        stage_entry = entry.clone()
        for keys, pair in original_groups[:11]:
            for action in keys: stage_entry.step(action)
            for action in pair: stage_entry.step(*action)
        stage_boundary = entry.clone()
        for keys, pair in original_groups[:17]:
            for action in keys: stage_boundary.step(action)
            for action in pair: stage_boundary.step(*action)
        boundary_key = state_key(stage_boundary)
        suffix_pairs = tuple(pair for _, pair in original_groups[14:17])
        for order in orders:
            node = stage_entry.clone(); candidate = []; trace = []
            for name in order:
                pair = stage_moves[name]
                alignment, searched = align(node, pair)
                trace.append((name, None if alignment is None else len(alignment), searched))
                if alignment is None: break
                for action in alignment: node.step(action); candidate.append(action)
                for action in pair: node.step(*action); candidate.append(action)
            else:
                for pair in suffix_pairs:
                    for action in pair: node.step(*action); candidate.append(action)
                same = state_key(node) == boundary_key
                if same:
                    for keys, pair in original_groups[17:]:
                        for action in keys: node.step(action); candidate.append(action)
                        for action in pair: node.step(*action); candidate.append(action)
                        if node.levels_completed >= 7: break
                print("ORDER_RESULT", order, len(candidate), same, node.levels_completed >= 7, trace, flush=True)
                if node.levels_completed >= 7: print("ORDER_PATH", candidate, flush=True)
                continue
            print("ORDER_RESULT", order, None, False, False, trace, flush=True)
        return
    original_boundary = entry.clone()
    for keys, pair in original_groups[:9]:
        for action in keys: original_boundary.step(action)
        for action in pair: original_boundary.step(*action)
    boundary_key = state_key(original_boundary)

    suffix_pairs = tuple(pair for _, pair in original_groups[4:9])
    for order in ORDERS:
        node = entry.clone()
        candidate = []
        trace = []
        for name in order:
            pair = MOVES[name]
            alignment, searched = align(node, pair)
            trace.append((name, None if alignment is None else len(alignment), searched))
            if alignment is None:
                break
            for action in alignment: node.step(action); candidate.append(action)
            for action in pair: node.step(*action); candidate.append(action)
        else:
            for pair in suffix_pairs:
                for action in pair: node.step(*action); candidate.append(action)
            same = state_key(node) == boundary_key
            if same:
                for keys, pair in original_groups[9:]:
                    for action in keys: node.step(action); candidate.append(action)
                    for action in pair: node.step(*action); candidate.append(action)
                    if node.levels_completed >= 7: break
            print(
                "ORDER_RESULT", order, len(candidate), same,
                node.levels_completed >= 7, trace, flush=True,
            )
            if node.levels_completed >= 7:
                print("ORDER_PATH", candidate, flush=True)
            continue
        print("ORDER_RESULT", order, None, False, False, trace, flush=True)


gkm_try.A.run_program("lf52", probe)

"""Compact legal-move frontier in level seven's final carrier region."""

import json
import os
from collections import deque

import gkm_try
from legs import _movable_bridge_board
from perception import arr, connected_components


MAX_STATES = int(os.environ.get("MAX_STATES", "800"))
MAX_DEPTH = int(os.environ.get("MAX_DEPTH", "24"))
ANY_EXIT = os.environ.get("ANY_EXIT") == "1"


def board(frame):
    slots, carriers, bridges, pegs = _movable_bridge_board(frame)
    fixed = {
        (blob.bbox[0] + 1, blob.bbox[1])
        for blob in connected_components(frame, colors=(15,))
        if blob.size == (4, 4) and blob.area >= 12
    }
    return slots, carriers, bridges, pegs, fixed


def legal(frame):
    slots, carriers, bridges, pegs, fixed = board(frame)
    occupied = bridges | pegs
    result = []
    for kind, sources in (("B", bridges), ("P", pegs)):
        for source in sorted(sources):
            for dr, dc in ((-12, 0), (12, 0), (0, -12), (0, 12)):
                destination = source[0] + dr, source[1] + dc
                midpoint = source[0] + dr // 2, source[1] + dc // 2
                if (
                    destination in slots | carriers
                    and destination not in occupied
                    and midpoint in occupied | fixed
                ):
                    result.append((kind, source, destination, midpoint))
    return tuple(result)


def frame_key(node):
    return arr(node.frame())[1:, :].tobytes()


def align(root, desired):
    queue = deque([(root.clone(), ())])
    seen = {frame_key(root)}
    while queue and len(seen) <= MAX_STATES:
        node, path = queue.popleft()
        if any((source, destination) == desired for _, source, destination, _ in legal(node.frame())):
            return path, len(seen)
        if len(path) >= MAX_DEPTH:
            continue
        for action in (1, 2, 3, 4):
            child = node.clone(); child.step(action)
            child_key = frame_key(child)
            if child_key in seen: continue
            seen.add(child_key)
            queue.append((child, path + (action,)))
    return None, len(seen)


def probe(env):
    with open("checkpoint.json") as checkpoint_file:
        path = json.load(checkpoint_file)["final_path"]
    for action in path[:331]: env.step(action)
    segment = path[331:476]
    coordinate_pairs = 0
    index = 0
    while index < len(segment) and coordinate_pairs < 17:
        action = segment[index]
        if isinstance(action, list):
            env.step(*action)
            if index + 1 < len(segment) and isinstance(segment[index + 1], list):
                index += 1; env.step(*segment[index]); coordinate_pairs += 1
        else:
            env.step(action)
        index += 1
    print("FINAL_ENTRY", board(env.frame()), legal(env.frame()))
    final_entry = env.clone()
    queue = deque([(env.clone(), ())])
    seen = set()
    while queue and len(seen) < 80:
        node, moves = queue.popleft()
        signature = (frozenset(board(node.frame())[2]), frozenset(board(node.frame())[3]))
        if signature in seen: continue
        seen.add(signature)
        print("FINAL_FRONTIER", moves, signature, legal(node.frame()))
        if len(moves) >= 4: continue
        for kind, source, destination, _ in legal(node.frame()):
            child = node.clone()
            child.step(6, source[1] + 1, source[0] + 1)
            child.step(6, destination[1] + 1, destination[0] + 1)
            queue.append((child, moves + ((kind, source, destination),)))

    alternate_moves = (
        ((36, 18), (36, 6)),
        ((42, 6), (30, 6)),
        ((36, 6), (24, 6)),
        ((30, 6), (18, 6)),
    )
    alternate = final_entry.clone(); actions = []
    for source, destination in alternate_moves:
        pair = (
            (6, source[1] + 1, source[0] + 1),
            (6, destination[1] + 1, destination[0] + 1),
        )
        for action in pair: alternate.step(*action); actions.append(action)
    if ANY_EXIT:
        search = deque([(alternate.clone(), ())])
        seen_keys = {frame_key(alternate)}
        found_depth = None
        found = []
        while search and len(seen_keys) <= MAX_STATES:
            candidate, key_path = search.popleft()
            if found_depth is not None and len(key_path) > found_depth:
                break
            novel = tuple(
                move for move in legal(candidate.frame())
                if (move[1], move[2]) not in {
                    ((18, 6), (30, 6)), ((24, 6), (36, 6))
                }
            )
            if novel:
                found_depth = len(key_path); found.append((key_path, novel, board(candidate.frame())[1:4]))
                continue
            if len(key_path) >= MAX_DEPTH: continue
            for action in (1, 2, 3, 4):
                child = candidate.clone(); child.step(action)
                child_key = frame_key(child)
                if child_key in seen_keys: continue
                seen_keys.add(child_key); search.append((child, key_path + (action,)))
        print("FINAL_ANY_EXIT", len(seen_keys), found_depth, found[:12])
        return
    first, first_seen = align(alternate, ((24, 54), (36, 54)))
    trace = [("unload", None if first is None else len(first), first_seen)]
    if first is not None:
        for action in first: alternate.step(action); actions.append(action)
        for action in ((6, 55, 25), (6, 55, 37)):
            alternate.step(*action); actions.append(action)
        second, second_seen = align(alternate, ((42, 54), (30, 54)))
        trace.append(("capture", None if second is None else len(second), second_seen))
        if second is not None:
            for action in second: alternate.step(action); actions.append(action)
            for action in ((6, 55, 43), (6, 55, 31)):
                alternate.step(*action); actions.append(action)
    print("FINAL_ALTERNATE", len(actions), alternate.levels_completed >= 7, trace, actions)


gkm_try.A.run_program("lf52", probe)

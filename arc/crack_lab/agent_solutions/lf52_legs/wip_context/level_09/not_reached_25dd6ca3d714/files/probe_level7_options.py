"""Enumerate shortest key alignments to legal level-7 piece moves."""

import json
import os
import sys
from collections import deque

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from legs import _lattice_step, _movable_bridge_board
from perception import arr, connected_components, safe_step


PREFIX = 331
KEYS = (1, 2, 3, 4)


def state_key(node):
    return arr(node.frame())[1:, :].tobytes()


def candidate_moves(frame):
    slots, carriers, detected_bridges, pegs = _movable_bridge_board(frame)
    bridges = set(detected_bridges)
    bridges.update(
        blob.top_left for blob in connected_components(frame, colors=(9,))
        if blob.size == (4, 4) and blob.area == 12
    )
    fixed = {
        (blob.bbox[0] + 1, blob.bbox[1])
        for blob in connected_components(frame, colors=(15,))
        if blob.size == (4, 4) and blob.area == 12
    }
    step = 6
    occupied = pegs | bridges
    destinations = slots | carriers
    found = []
    for kind, pieces in (("peg", pegs), ("bridge", bridges)):
        for source in sorted(pieces):
            for dr, dc in ((-step, 0), (step, 0), (0, -step), (0, step)):
                midpoint = (source[0] + dr, source[1] + dc)
                destination = (source[0] + 2 * dr, source[1] + 2 * dc)
                support = occupied | fixed
                if (
                    midpoint in support
                    and destination in destinations
                    and destination not in occupied
                ):
                    found.append((kind, source, destination))
    return tuple(found)


def observe(env):
    target_level = int(os.environ.get("TARGET_LEVEL", "7"))
    prefix_length = {6: 238, 7: PREFIX, 8: 476}[target_level]
    with open("checkpoint.json") as stream:
        path = json.load(stream)["final_path"]
    for action in path[:prefix_length]:
        safe_step(env, action)

    cross_first = os.environ.get("CROSS_FIRST") == "1"
    bridge_first = os.environ.get("BRIDGE_FIRST") == "1"
    if cross_first:
        cross_stage = int(os.environ.get("CROSS_STAGE", "4"))
        for action in (3, 3, 1, 1, 3, 3, 3):
            safe_step(env, action)
        for action in ((6, 7, 13), (6, 7, 25)):
            safe_step(env, action)
        if cross_stage >= 2:
            for action in (4, 4, 4, 2, 2, 4, 4, 4, 2):
                safe_step(env, action)
            for action in ((6, 43, 43), (6, 43, 55)):
                safe_step(env, action)
        if cross_stage >= 3:
            for action in (1, 1, 1):
                safe_step(env, action)
            for action in ((6, 43, 13), (6, 43, 25)):
                safe_step(env, action)
        if cross_stage >= 4:
            for action in (3, 3, 3, 2, 2, 3, 3, 2):
                safe_step(env, action)
            for action in ((6, 13, 43), (6, 13, 55)):
                safe_step(env, action)
    if bridge_first:
        for action in (3, 3, 1, 1, 4, 4, 4):
            safe_step(env, action)
        for action in ((6, 43, 13), (6, 43, 25)):
            safe_step(env, action)
        if os.environ.get("BRIDGE_FIRST_UNLOAD") == "1":
            for action in (3, 3, 3, 2, 2, 3, 3, 2):
                safe_step(env, action)
            for action in ((6, 13, 43), (6, 13, 55)):
                safe_step(env, action)
            if os.environ.get("BRIDGE_FIRST_LOAD_PEG") == "1":
                for action in (1, 4, 4, 1, 1, 3, 3, 3):
                    safe_step(env, action)
                for action in ((6, 7, 13), (6, 7, 25)):
                    safe_step(env, action)
                if os.environ.get("BRIDGE_FIRST_UNLOAD_PEG") == "1":
                    for action in (4, 4, 4, 2, 2, 4, 4, 4, 2):
                        safe_step(env, action)
                    for action in ((6, 43, 43), (6, 43, 55)):
                        safe_step(env, action)
                    if os.environ.get("L7_SWAPPED_RING") == "1":
                        for move in (
                            ((54, 12), (54, 24)),
                            ((54, 24), (54, 36)),
                            ((54, 36), (54, 48)),
                            ((54, 42), (54, 54)),
                            ((54, 4), (54, 16)),
                        ):
                            source, destination = move
                            for action in (
                                (6, source[1] + 1, source[0] + 1),
                                (6, destination[1] + 1,
                                 destination[0] + 1),
                            ):
                                safe_step(env, action)
                        if os.environ.get("L7_SWAPPED_LOAD") == "1":
                            for action in (
                                1, 3, 3, 3, 2, 2, 3, 3,
                                2, 2, 3, 2, 2, 4, 4, 2,
                            ):
                                safe_step(env, action)
                            for move in (
                                ((54, 10), (54, 22)),
                                ((54, 16), (54, 28)),
                            ):
                                source, destination = move
                                for action in (
                                    (6, source[1] + 1, source[0] + 1),
                                    (6, destination[1] + 1,
                                     destination[0] + 1),
                                ):
                                    safe_step(env, action)

    completed_macros = int(os.environ.get("AFTER_MACROS", "0"))
    coordinate_actions = 0
    for action in (() if bridge_first or cross_first else
                   path[prefix_length:]):
        if coordinate_actions >= 2 * completed_macros:
            break
        safe_step(env, action)
        if isinstance(action, list):
            coordinate_actions += 1

    if os.environ.get("ALT_FINAL_PEG_FIRST") == "1":
        for action in ((6, 7, 43), (6, 19, 43)):
            safe_step(env, action)
        if os.environ.get("ALT_FINAL_PEG_TWICE") == "1":
            for action in ((6, 19, 43), (6, 31, 43)):
                safe_step(env, action)

    if os.environ.get("L6_SHORT_ENTRY") == "1":
        for action in (4, 4, 4, 4, 4, 4, 4, 1, 1):
            safe_step(env, action)
        for action in ((6, 29, 31), (6, 29, 19)):
            safe_step(env, action)

    if os.environ.get("L6_BRIDGE_ENTRY") == "1":
        for action in (4, 4, 4, 4, 4, 4, 1, 1):
            safe_step(env, action)
        for action in ((6, 35, 31), (6, 35, 19)):
            safe_step(env, action)

    if os.environ.get("L6_STAGE15_DOWN") == "1":
        for action in ((6, 3, 19), (6, 3, 31)):
            safe_step(env, action)

    if os.environ.get("L6_STAGE21_BRIDGE_RIGHT") == "1":
        for action in (2, 2, 4, 4, 1, 1, 1, 1):
            safe_step(env, action)
        for action in ((6, 39, 19), (6, 51, 19)):
            safe_step(env, action)
        if os.environ.get("L6_STAGE21_PEG_RIGHT") == "1":
            for action in ((6, 45, 19), (6, 57, 19)):
                safe_step(env, action)
            if os.environ.get("L6_STAGE21_PEG_DOWN") == "1":
                for action in ((6, 57, 19), (6, 57, 31)):
                    safe_step(env, action)
                if os.environ.get("L6_STAGE21_PEG_LEFT") == "1":
                    for action in (2, 2):
                        safe_step(env, action)
                    for action in ((6, 57, 31), (6, 45, 31)):
                        safe_step(env, action)
                    if os.environ.get("L6_STAGE21_PEG_LEFT2") == "1":
                        for action in (2, 2, 3, 3, 1, 1):
                            safe_step(env, action)
                        for action in ((6, 45, 31), (6, 33, 31)):
                            safe_step(env, action)
                        if os.environ.get("L6_STAGE21_BRIDGE_UP") == "1":
                            for action in ((6, 39, 31), (6, 39, 19),
                                           (6, 33, 31), (6, 33, 43)):
                                safe_step(env, action)
                        elif os.environ.get("L6_STAGE21_CAPTURE") == "1":
                            for action in ((6, 33, 31), (6, 33, 43)):
                                safe_step(env, action)

    if os.environ.get("L8_SHORT12") == "1":
        for action in (3, 2, 2, 2, 2, 4, 4, 4, 2):
            safe_step(env, action)
        for action in ((6, 55, 43), (6, 43, 43)):
            safe_step(env, action)
        if os.environ.get("L8_SHORT_MIDDLE") == "1":
            for source, destination in (
                ((42, 48), (42, 36)),
                ((42, 42), (42, 30)),
                ((42, 36), (42, 24)),
                ((36, 24), (48, 24)),
            ):
                for action in (
                    (6, source[1] + 1, source[0] + 1),
                    (6, destination[1] + 1, destination[0] + 1),
                ):
                    safe_step(env, action)
            if os.environ.get("L8_SHORT_FIX") == "1":
                for action in (3, 3, 3):
                    safe_step(env, action)
                for action in ((6, 25, 37), (6, 25, 49)):
                    safe_step(env, action)

    if os.environ.get("L8_FAST17") == "1":
        for action in ((6, 25, 43), (6, 37, 43)):
            safe_step(env, action)
        if os.environ.get("L8_FAST_BRIDGES") == "1":
            for source, destination in (
                ((42, 30), (42, 42)),
                ((42, 36), (42, 48)),
            ):
                for action in (
                    (6, source[1] + 1, source[0] + 1),
                    (6, destination[1] + 1, destination[0] + 1),
                ):
                    safe_step(env, action)

    trace_keys = tuple(int(value) for value in
                       os.environ.get("TRACE_KEYS", "").split(",") if value)
    if trace_keys:
        print("TRACE_OPTION", {"index": 0,
                               "moves": candidate_moves(env.frame())})
        for index, action in enumerate(trace_keys, 1):
            safe_step(env, action)
            print("TRACE_OPTION", {"index": index, "action": action,
                                   "moves": candidate_moves(env.frame())})
        return

    root = env.clone()
    max_depth = int(os.environ.get("MAX_DEPTH", "12"))
    max_states = int(os.environ.get("MAX_STATES", "500"))
    queue = deque([(root, ())])
    seen = {state_key(root)}
    options = {}
    while queue and len(seen) <= max_states:
        node, path = queue.popleft()
        for move in candidate_moves(node.frame()):
            options.setdefault(move, path)
        if len(path) >= max_depth:
            continue
        for action in KEYS:
            child = node.clone()
            safe_step(child, action)
            key = state_key(child)
            if key not in seen:
                seen.add(key)
                queue.append((child, path + (action,)))

    print("OPTION_SEARCH", {"bridge_first": bridge_first,
                            "cross_first": cross_first,
                            "target_level": target_level,
                            "after_macros": completed_macros,
                            "states": len(seen), "remaining": len(queue),
                            "max_depth": max_depth})
    for move, path in sorted(options.items(), key=lambda item:
                             (len(item[1]), item[0])):
        print("OPTION", {"keys": path, "move": move})


levels, path, error = arena.run_program("lf52", observe)
print("PROBE_RESULT", {"levels": levels, "moves": len(path),
                       "error": str(error)})

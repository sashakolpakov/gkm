"""Enumerate validated piece moves in a bounded movable-carrier key orbit."""

from collections import deque
import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from legs import _movable_bridge_board
from perception import arr, safe_step


BOUNDARIES = (0, 8, 42, 87, 149, 238, 331, 476, 544)
KEYS = (1, 2, 3, 4, 7)


def state_key(node):
    return int(node.levels_completed), arr(node.frame())[1:, :].tobytes()


def clicks(source, destination):
    return (
        (6, source[1] + 1, source[0] + 1),
        (6, destination[1] + 1, destination[0] + 1),
    )


def validated_moves(node, base_level):
    slots, carriers, bridges, pegs = _movable_bridge_board(node.frame())
    destinations = slots | carriers
    moves = []
    for kind, pieces in (("bridge", bridges), ("peg", pegs)):
        for source in sorted(pieces):
            for destination in sorted(destinations):
                distance = (
                    abs(source[0] - destination[0]),
                    abs(source[1] - destination[1]),
                )
                if distance not in ((0, 12), (12, 0)):
                    continue
                child = node.clone()
                for action in clicks(source, destination):
                    safe_step(child, action)
                if child.levels_completed > base_level:
                    moves.append(((kind, source, destination), ("reward",)))
                    continue
                _, after_carriers, after_bridges, after_pegs = _movable_bridge_board(
                    child.frame()
                )
                after_pieces = after_bridges if kind == "bridge" else after_pegs
                if source not in after_pieces and destination in after_pieces:
                    moves.append((
                        (kind, source, destination),
                        (
                            tuple(sorted(after_carriers)),
                            tuple(sorted(after_bridges)),
                            tuple(sorted(after_pegs)),
                        ),
                    ))
    return tuple(moves)


def level_segment(level, checkpoint):
    candidate_name = os.environ.get("CANDIDATE")
    if candidate_name:
        with open(candidate_name) as stream:
            return json.load(stream)
    return checkpoint[BOUNDARIES[level - 1]:BOUNDARIES[level]]


def observe(env):
    level = int(os.environ.get("TARGET_LEVEL", "7"))
    after_macros = int(os.environ.get("AFTER_MACROS", "0"))
    max_depth = int(os.environ.get("MAX_DEPTH", "16"))
    max_states = int(os.environ.get("MAX_STATES", "700"))
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)["final_path"]
    for action in checkpoint[:BOUNDARIES[level - 1]]:
        safe_step(env, action)
    segment = level_segment(level, checkpoint)
    coordinate_actions = 0
    for action in segment:
        if coordinate_actions >= 2 * after_macros:
            break
        safe_step(env, action)
        if isinstance(action, list):
            coordinate_actions += 1

    root = env.clone()
    base_level = int(root.levels_completed)
    queue = deque([(root, ())])
    seen = {state_key(root)}
    options = {}
    while queue and len(seen) <= max_states:
        node, path = queue.popleft()
        for move, child_state in validated_moves(node, base_level):
            options.setdefault((move, child_state), path)
        if len(path) >= max_depth:
            continue
        for action in KEYS:
            child = node.clone()
            safe_step(child, action)
            key = state_key(child)
            if key not in seen:
                seen.add(key)
                queue.append((child, path + (action,)))

    print("MOVABLE_OPTIONS", {
        "level": level,
        "after_macros": after_macros,
        "states": len(seen),
        "remaining": len(queue),
    })
    for (move, child_state), path in sorted(
        options.items(), key=lambda item: (len(item[1]), item[0])
    ):
        print("OPTION", {
            "keys": path,
            "move": move,
            "child": child_state,
        })


levels, path, error = arena.run_program("lf52", observe)
print("PROBE_RESULT", {
    "levels": levels,
    "moves": len(path),
    "error": str(error),
})

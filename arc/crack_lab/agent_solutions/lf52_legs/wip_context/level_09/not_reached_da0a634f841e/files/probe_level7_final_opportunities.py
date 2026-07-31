"""Enumerate shortest carrier paths to each legal final-board lattice move."""

from collections import deque
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from legs import _bridge_carrier_state, _movable_bridge_board


FINAL_ENTRY_INDEX = int(os.environ.get("OPPORTUNITY_ENTRY", "446"))
MAX_KEY_DEPTH = int(os.environ.get("OPPORTUNITY_DEPTH", "6"))
MAX_STATES = int(os.environ.get("OPPORTUNITY_STATES", "200"))
SETUP = os.environ.get("OPPORTUNITY_SETUP", "")


def physical_key(env):
    return tuple(frozenset(part) for part in _movable_bridge_board(env.frame()))


def legal_moves(env):
    slots, carriers, bridges, pegs = _movable_bridge_board(env.frame())
    static_bridges = _bridge_carrier_state(env.frame())[3]
    destinations = slots | carriers
    occupied = pegs | bridges | static_bridges
    moves = []
    for kind, pieces in (("peg", pegs), ("bridge", bridges)):
        for source in sorted(pieces):
            for dr, dc in ((-6, 0), (6, 0), (0, -6), (0, 6)):
                midpoint = (source[0] + dr, source[1] + dc)
                destination = (source[0] + 2 * dr, source[1] + 2 * dc)
                if (
                    midpoint not in occupied
                    or destination not in destinations
                    or destination in occupied
                ):
                    continue
                moves.append((kind, source, destination))
    return tuple(moves)


def probe(env):
    with open("checkpoint.json") as checkpoint_file:
        campaign = json.load(checkpoint_file)["final_path"]
    for action in campaign[:FINAL_ENTRY_INDEX]:
        if isinstance(action, list):
            env.step(*action)
        else:
            env.step(action)
    if SETUP == "bridge_first":
        for action in (3, 3, 1, 1, 4, 4, 4):
            env.step(action)
        env.step(6, 43, 13)
        env.step(6, 43, 25)
    elif SETUP == "bridge_first_to_12":
        for action in (
            3, 3, 1, 1, 4, 4, 4,
            (6, 43, 13), (6, 43, 25),
            3, 3, 3, 2, 2, 3, 3, 2,
            (6, 13, 43), (6, 13, 55),
        ):
            if isinstance(action, tuple):
                env.step(*action)
            else:
                env.step(action)
    elif SETUP == "bridge_first_to_48":
        for action in (
            3, 3, 1, 1, 4, 4, 4,
            (6, 43, 13), (6, 43, 25),
            3, 3, 3, 2, 2, 3, 3, 2,
            (6, 13, 43), (6, 13, 55),
            (6, 13, 55), (6, 25, 55),
            (6, 25, 55), (6, 37, 55),
            (6, 37, 55), (6, 49, 55),
        ):
            if isinstance(action, tuple):
                env.step(*action)
            else:
                env.step(action)
    elif SETUP == "swapped_peg_loaded":
        for action in (
            3, 3, 1, 1, 4, 4, 4,
            (6, 43, 13), (6, 43, 25),
            3, 3, 3, 2, 2, 3, 3, 2,
            (6, 13, 43), (6, 13, 55),
            1, 4, 4, 1, 1, 3, 3, 3,
            (6, 7, 13), (6, 7, 25),
        ):
            if isinstance(action, tuple):
                env.step(*action)
            else:
                env.step(action)
    elif SETUP == "swapped_endpoints":
        for action in (
            3, 3, 1, 1, 4, 4, 4,
            (6, 43, 13), (6, 43, 25),
            3, 3, 3, 2, 2, 3, 3, 2,
            (6, 13, 43), (6, 13, 55),
            1, 4, 4, 1, 1, 3, 3, 3,
            (6, 7, 13), (6, 7, 25),
            3, 3, 3, 2, 2, 4, 4, 4, 2,
            (6, 43, 43), (6, 43, 55),
        ):
            if isinstance(action, tuple):
                env.step(*action)
            else:
                env.step(action)
    elif SETUP == "swapped_bridge_24":
        for action in (
            3, 3, 1, 1, 4, 4, 4,
            (6, 43, 13), (6, 43, 25),
            3, 3, 3, 2, 2, 3, 3, 2,
            (6, 13, 43), (6, 13, 55),
            1, 4, 4, 1, 1, 3, 3, 3,
            (6, 7, 13), (6, 7, 25),
            3, 3, 3, 2, 2, 4, 4, 4, 2,
            (6, 43, 43), (6, 43, 55),
            (6, 13, 55), (6, 25, 55),
        ):
            if isinstance(action, tuple):
                env.step(*action)
            else:
                env.step(action)
    elif SETUP == "swapped_bridge_36":
        for action in (
            3, 3, 1, 1, 4, 4, 4,
            (6, 43, 13), (6, 43, 25),
            3, 3, 3, 2, 2, 3, 3, 2,
            (6, 13, 43), (6, 13, 55),
            1, 4, 4, 1, 1, 3, 3, 3,
            (6, 7, 13), (6, 7, 25),
            3, 3, 3, 2, 2, 4, 4, 4, 2,
            (6, 43, 43), (6, 43, 55),
            (6, 13, 55), (6, 25, 55),
            (6, 25, 55), (6, 37, 55),
        ):
            if isinstance(action, tuple):
                env.step(*action)
            else:
                env.step(action)
    root = env.clone()
    queue = deque([(root, ())])
    seen = {physical_key(root)}
    opportunities = {}
    while queue and len(seen) <= MAX_STATES:
        state, path = queue.popleft()
        for move in legal_moves(state):
            current = opportunities.get(move)
            board = _movable_bridge_board(state.frame())
            if current is None or len(path) < len(current["keys"]):
                opportunities[move] = {
                    "keys": path,
                    "board": board,
                }
        if len(path) >= MAX_KEY_DEPTH:
            continue
        for action in (1, 2, 3, 4):
            child = state.clone()
            child.step(action)
            key = physical_key(child)
            if key in seen:
                continue
            seen.add(key)
            queue.append((child, path + (action,)))
    print("RESULT", {
        "states": len(seen),
        "opportunities": [
            {
                "move": move,
                "keys": value["keys"],
                "board": value["board"],
            }
            for move, value in sorted(
                opportunities.items(),
                key=lambda item: (len(item[1]["keys"]), item[0]),
            )
        ],
    })


levels, path, error = arena.run_program("lf52", probe)
print("HARNESS", {
    "levels": levels,
    "moves": len(path),
    "error": str(error) if error else None,
})

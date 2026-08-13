"""Inspect and test direct finishes at the close level-9 peg encounter."""

import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from legs import (
    _bridge_carrier_moves,
    _bridge_carrier_state,
    _movable_bridge_board,
)


CONTEXT_INDEX = 62


def play(env, action):
    if isinstance(action, list):
        env.step(*action)
    else:
        env.step(action)


def probe(env):
    with open("checkpoint.json") as checkpoint_file:
        prefix = json.load(checkpoint_file)["final_path"]
    with open("level9_candidate_102.json") as candidate_file:
        candidate = json.load(candidate_file)
    for action in prefix:
        play(env, action)
    node = env.clone()
    for action in candidate[:CONTEXT_INDEX]:
        play(node, action)
    print("STATE", {
        "bridge": _bridge_carrier_state(node.frame()),
        "movable": _movable_bridge_board(node.frame()),
        "legal": _bridge_carrier_moves(node.frame()),
    })
    points = (
        ((24, 16), (36, 16)),
        ((24, 16), (24, 28)),
        ((24, 16), (36, 28)),
        ((36, 22), (36, 10)),
        ((36, 22), (36, 34)),
        ((36, 22), (24, 22)),
    )
    for source, destination in points:
        child = node.clone()
        before = _bridge_carrier_state(child.frame())
        child.step(6, source[1] + 1, source[0] + 1)
        child.step(6, destination[1] + 1, destination[0] + 1)
        after = _bridge_carrier_state(child.frame())
        print("TRY", {
            "move": (source, destination),
            "changed": after != before,
            "level": child.levels_completed,
            "after_pegs": sorted(after[1]),
            "after_legal": _bridge_carrier_moves(child.frame()),
        })
    left = node.clone()
    for turns in range(1, 11):
        left.step(3)
        state = _bridge_carrier_state(left.frame())
        print("LEFT", {
            "turns": turns,
            "level": left.levels_completed,
            "pegs": sorted(state[1]),
            "bridges": sorted(state[3]),
            "row36_slots": sorted(
                position for position in state[0] if position[0] == 36
            ),
            "legal": _bridge_carrier_moves(left.frame()),
        })
    reverse = node.clone()
    for _ in range(7):
        reverse.step(3)
    reverse.step(6, 59, 25)
    reverse.step(6, 47, 25)
    print("REVERSE_START", {
        "level": reverse.levels_completed,
        "state": _bridge_carrier_state(reverse.frame()),
        "legal": _bridge_carrier_moves(reverse.frame()),
    })
    for turns in range(1, 11):
        reverse.step(3)
        state = _bridge_carrier_state(reverse.frame())
        print("REVERSE_LEFT", {
            "turns": turns,
            "level": reverse.levels_completed,
            "pegs": sorted(state[1]),
            "bridges": sorted(state[3]),
            "row36_slots": sorted(
                position for position in state[0] if position[0] == 36
            ),
            "legal": _bridge_carrier_moves(reverse.frame()),
        })


levels, path, error = arena.run_program("lf52", probe)
print("HARNESS", {
    "levels": levels,
    "moves": len(path),
    "error": str(error) if error else None,
})

"""Compact symbolic snapshots at each transition in the level-9 relay."""

import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from legs import _bridge_carrier_state, _movable_bridge_board
from perception import color_counts


def play(env, action):
    if isinstance(action, list):
        env.step(*action)
    else:
        env.step(action)


def snapshot(env, index, label):
    slots, carriers, bridges, pegs = _movable_bridge_board(env.frame())
    b_slots, b_pegs, b_carriers, b_bridges, borders, selected = (
        _bridge_carrier_state(env.frame())
    )
    print("STAGE", {
        "at": index,
        "label": label,
        "level": env.levels_completed,
        "movable": {
            "slots": len(slots),
            "carriers": sorted(carriers),
            "bridges": sorted(bridges),
            "pegs": sorted(pegs),
        },
        "bridge": {
            "slots": len(b_slots),
            "pegs": sorted(b_pegs),
            "carriers": sorted(b_carriers),
            "bridges": sorted(b_bridges),
            "borders": sorted(borders),
            "selected": selected,
        },
        "counts": {
            color: color_counts(env.frame()).get(color, 0)
            for color in (1, 5, 8, 9, 10, 12, 14, 15)
        },
    })


def probe(env):
    with open("checkpoint.json") as checkpoint_file:
        prefix = json.load(checkpoint_file)["final_path"]
    with open("level9_candidate_102.json") as candidate_file:
        candidate = json.load(candidate_file)
    for action in prefix:
        play(env, action)
    node = env.clone()
    snapshot(node, 0, "entry")
    for index, action in enumerate(candidate, 1):
        play(node, action)
        next_action = candidate[index] if index < len(candidate) else None
        if (
            index == 28
            or isinstance(action, int)
            or (isinstance(action, list) and isinstance(next_action, int))
            or node.levels_completed > 8
        ):
            snapshot(node, index, f"after {action}")


levels, path, error = arena.run_program("lf52", probe)
print("HARNESS", {
    "levels": levels,
    "moves": len(path),
    "error": str(error) if error else None,
})

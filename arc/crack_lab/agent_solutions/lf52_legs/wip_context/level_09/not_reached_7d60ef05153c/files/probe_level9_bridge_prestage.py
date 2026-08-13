"""Verify bridge-loaded carrier motion before level 9's peg entry."""

import json

import gkm_try

from perception import safe_step
from probe_level9 import (
    key_effects,
    legal_piece_moves,
    pieces,
    symbolic_carrier_entry,
)


def probe(env):
    with open("checkpoint.json") as checkpoint_file:
        prefix = json.load(checkpoint_file)["final_path"]
    for action in prefix:
        safe_step(env, action)
    solution, searched = symbolic_carrier_entry(env.frame(), "bridge")
    print("L9_PRESTAGE_ENTRY", searched, solution)
    if solution is None:
        return
    loaded = env.clone(); actions = []
    for _, source, destination in solution:
        pair = (
            (6, source[1] + 1, source[0] + 1),
            (6, destination[1] + 1, destination[0] + 1),
        )
        for action in pair:
            safe_step(loaded, action); actions.append(action)
    print("L9_PRESTAGE_LOADED", len(actions), pieces(loaded.frame()), key_effects(loaded))

    for direction in (3, 4):
        for count in range(1, 8):
            node = loaded.clone(); key_path = (direction,) * count
            for action in key_path:
                safe_step(node, action)
            print(
                "L9_PRESTAGE_SHIFT", key_path, pieces(node.frame()),
                key_effects(node), legal_piece_moves(node),
            )


gkm_try.A.run_program("lf52", probe)

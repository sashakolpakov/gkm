"""Deterministic carrier frontiers for promising crossed final-board branches."""

import json

import gkm_try

from perception import arr, safe_step
from probe_level7_crossed_keys import compact, legal_moves, variants
from probe_level7_crossed_final_frontier import TO_FINAL, apply_move


LEVEL_START = 331

BRIDGE_RIGHT = ("B", (42, 6), (42, 18))
BRIDGE_RIGHT2 = ("B", (42, 18), (42, 30))
PEG_LEFT = ("P", (36, 18), (36, 6))
PEG_RIGHT = ("P", (36, 18), (36, 30))
BRIDGE_UP = ("B", (42, 6), (30, 6))
PEG_UP = ("P", (36, 6), (24, 6))
BRIDGE_UP2 = ("B", (30, 6), (18, 6))


def probe(env):
    with open("checkpoint.json") as checkpoint_file:
        prefix = json.load(checkpoint_file)["final_path"]
    for action in prefix[:LEVEL_START]:
        safe_step(env, action)
    for action in TO_FINAL:
        safe_step(env, action)
    root = env.clone()

    coordinate_paths = (
        (),
        (PEG_RIGHT,),
        (BRIDGE_RIGHT,),
        (BRIDGE_RIGHT, BRIDGE_RIGHT2),
        (PEG_LEFT,),
        (PEG_LEFT, BRIDGE_UP),
        (PEG_LEFT, BRIDGE_UP, PEG_UP),
        (PEG_LEFT, BRIDGE_UP, PEG_UP, BRIDGE_UP2),
    )
    bases = (
        (2, 3, 3, 3, 2, 2),
        (1, 1, 4, 4, 4, 2, 2, 4, 2),
        (1, 3, 1, 3, 3, 2, 4, 2, 2),
        (1, 3, 3, 1, 1, 4, 4, 1, 1, 4, 4, 4, 2),
        (3, 3, 3, 2, 2, 4, 4, 2),
        (1, 3, 3, 1, 1, 4, 1, 1, 4, 4, 4),
    )
    key_paths = {()}
    for base in bases:
        key_paths.update(variants(base))
        key_paths.update(base[:cut] for cut in range(1, len(base)))
    for action in (1, 2, 3, 4):
        key_paths.update((action,) * count for count in range(1, 13))

    for coordinate_path in coordinate_paths:
        branch = root.clone()
        for move in coordinate_path:
            apply_move(branch, move)
        observations = {}; initial = compact(branch)
        for key_path in sorted(key_paths, key=lambda value: (len(value), value)):
            node = branch.clone()
            for action in key_path:
                safe_step(node, action)
            signature = arr(node.frame())[1:, :].tobytes()
            moves = legal_moves(node)
            item = key_path, compact(node), moves
            if signature not in observations or len(key_path) < len(observations[signature][0]):
                observations[signature] = item
        useful = sorted(
            (
                (len(key_path), key_path, state, moves)
                for key_path, state, moves in observations.values()
                if moves and (key_path == () or state != initial or moves != legal_moves(branch))
            ),
            key=lambda item: (item[0], item[1]),
        )
        print("L7_CROSS_FINAL_KEYS", coordinate_path, len(observations), len(useful))
        for item in useful:
            print("L7_CROSS_FINAL_KEY_STATE", coordinate_path, item)


gkm_try.A.run_program("lf52", probe)

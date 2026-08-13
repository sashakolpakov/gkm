"""Deterministic continuation after the crossed level-7 peg unload."""

import json

import gkm_try

from perception import arr, safe_step
from probe_level7_crossed_middle import CROSS_PREFIX
from probe_level7_crossed_keys import compact, legal_moves, variants


LEVEL_START = 331
PEG_TRANSPORT = (1, 3, 3, 1, 1, 4, 1, 1, 4, 4, 4)
BRIDGE_LOAD_ALIGN = (3, 3, 3, 2, 2, 4, 4, 2)
PEG_UNLOAD = ((6, 35, 25), (6, 47, 25))
BRIDGE_LOAD = ((6, 29, 55), (6, 29, 43))


def probe(env):
    with open("checkpoint.json") as checkpoint_file:
        prefix = json.load(checkpoint_file)["final_path"]
    for action in prefix[:LEVEL_START]:
        safe_step(env, action)
    for action in CROSS_PREFIX + PEG_TRANSPORT + PEG_UNLOAD + BRIDGE_LOAD_ALIGN + BRIDGE_LOAD:
        safe_step(env, action)
    root = env.clone()
    print("L7_CROSS_FOLLOW_ROOT", compact(root), legal_moves(root))

    bases = (
        (1, 3, 3, 1, 1, 4, 4, 1, 1, 4, 4, 4, 2),
        (1, 3, 3, 1, 1, 4, 1, 1, 4, 4, 4),
        (3, 3, 3, 2, 2, 4, 4, 2),
        (1, 3, 3, 1, 1, 4, 4, 4, 2),
        (1, 4, 4, 1, 1, 4, 4, 4),
    )
    paths = set()
    for base in bases:
        paths.update(variants(base))
        paths.update(base[:cut] for cut in range(1, len(base)))
    for action in (1, 2, 3, 4):
        paths.update((action,) * count for count in range(1, 13))

    observations = {}
    for path in sorted(paths, key=lambda value: (len(value), value)):
        node = root.clone()
        for action in path:
            safe_step(node, action)
        key = arr(node.frame())[1:, :].tobytes()
        moves = legal_moves(node)
        if key not in observations or len(path) < len(observations[key][0]):
            observations[key] = path, compact(node), moves
    useful = sorted(
        ((len(path), path, state, moves) for path, state, moves in observations.values() if moves),
        key=lambda item: (item[0], item[1]),
    )
    print("L7_CROSS_FOLLOW", len(paths), len(observations), len(useful))
    for item in useful:
        print("L7_CROSS_FOLLOW_STATE", item)


if __name__ == "__main__":
    gkm_try.A.run_program("lf52", probe)

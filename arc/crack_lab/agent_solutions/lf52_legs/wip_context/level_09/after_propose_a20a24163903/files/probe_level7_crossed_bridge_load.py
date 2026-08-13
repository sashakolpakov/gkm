"""Load the crossed final-board bridge into the central carrier."""

import json

import gkm_try

from perception import arr, safe_step
from probe_level7_crossed_keys import compact, legal_moves, variants
from probe_level7_crossed_final_frontier import TO_FINAL, apply_move


LEVEL_START = 331
SETUP = (
    ("P", (36, 18), (36, 30)),
    ("B", (42, 6), (42, 18)),
    ("B", (42, 18), (42, 30)),
)
ALIGN = (2, 3, 3, 3, 2, 2)
LOAD = ("B", (42, 30), (30, 30))


def probe(env):
    with open("checkpoint.json") as checkpoint_file:
        prefix = json.load(checkpoint_file)["final_path"]
    for action in prefix[:LEVEL_START]:
        safe_step(env, action)
    for action in TO_FINAL:
        safe_step(env, action)
    for move in SETUP:
        apply_move(env, move)
    for action in ALIGN:
        safe_step(env, action)
    print("L7_CROSS_BRIDGE_PRELOAD", compact(env), legal_moves(env))
    if LOAD not in legal_moves(env):
        return
    apply_move(env, LOAD); root = env.clone()
    print("L7_CROSS_BRIDGE_LOADED", compact(root), legal_moves(root))

    bases = (
        (1, 1, 4, 4, 4, 2, 2, 4, 2),
        (1, 3, 1, 3, 3, 2, 4, 2, 2),
        (2, 3, 3, 3, 2, 2),
        (1, 3, 3, 1, 1, 4, 4, 1, 1, 4, 4, 4, 2),
    )
    paths = {()}
    for base in bases:
        paths.update(variants(base))
        paths.update(base[:cut] for cut in range(1, len(base)))
    observations = {}; frontier = {}
    for path in sorted(paths, key=lambda value: (len(value), value)):
        node = root.clone()
        for action in path:
            safe_step(node, action)
        signature = arr(node.frame())[1:, :].tobytes()
        if signature in observations:
            continue
        observations[signature] = path
        for move in legal_moves(node):
            old = frontier.get(move)
            item = len(path), path, compact(node)
            if old is None or item[0] < old[0]:
                frontier[move] = item
    print("L7_CROSS_BRIDGE_FRONTIER", len(paths), len(observations), tuple(sorted(frontier.items())))


gkm_try.A.run_program("lf52", probe)

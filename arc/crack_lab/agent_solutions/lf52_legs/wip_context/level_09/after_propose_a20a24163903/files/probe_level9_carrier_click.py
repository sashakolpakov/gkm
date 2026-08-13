"""Test coordinate control of empty and loaded carrier rail destinations."""

import json

import gkm_try

from perception import safe_step
from probe_level9 import pieces, symbolic_solution


def probe_paths(root, label):
    source = (6, 43, 37)
    targets = tuple((6, column + 1, 37) for column in (48, 54, 60))
    for path in tuple((target,) for target in targets) + tuple((source, target) for target in targets):
        node = root.clone()
        for action in path:
            safe_step(node, action)
        print("L9_CARRIER_CLICK", label, path, pieces(node.frame()), int(node.levels_completed))


def probe(env):
    with open("checkpoint.json") as checkpoint_file:
        prefix = json.load(checkpoint_file)["final_path"]
    for action in prefix:
        safe_step(env, action)
    probe_paths(env, "empty")

    solution, _ = symbolic_solution(env.frame())
    loaded = env.clone()
    for _, source, destination in solution:
        safe_step(loaded, (6, source[1] + 1, source[0] + 1))
        safe_step(loaded, (6, destination[1] + 1, destination[0] + 1))
    probe_paths(loaded, "loaded")


gkm_try.A.run_program("lf52", probe)

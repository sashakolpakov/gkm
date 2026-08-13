"""Verify selected-piece effects on empty carrier movement at level 9."""

import json

import gkm_try

from perception import safe_step
from probe_level9 import pieces, symbolic_solution


def compact(node):
    state = pieces(node.frame())
    return state["carriers"], state["pegs"], state["selected"]


def run_paths(root, label, paths):
    for path in paths:
        node = root.clone()
        for action in path:
            safe_step(node, action)
        print("L9_SELECTION", label, path, compact(node), int(node.levels_completed))


def probe(env):
    with open("checkpoint.json") as checkpoint_file:
        prefix = json.load(checkpoint_file)["final_path"]
    for action in prefix:
        safe_step(env, action)
    run_paths(env, "root", (
        (4,), ((6, 19, 43),), ((6, 19, 43), 4),
        ((6, 25, 43), 4),
    ))

    solution, _ = symbolic_solution(env.frame())
    pre_load = env.clone()
    for _, source, destination in solution[:-1]:
        safe_step(pre_load, (6, source[1] + 1, source[0] + 1))
        safe_step(pre_load, (6, destination[1] + 1, destination[0] + 1))
    source = (6, solution[-1][1][1] + 1, solution[-1][1][0] + 1)
    run_paths(pre_load, "preload", (
        (), (4,), (4, 4), (source,), (source, 4), (source, 3),
        (source, 4, 3), (source, 1), (source, 2),
    ))


gkm_try.A.run_program("lf52", probe)

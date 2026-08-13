"""Test whether undo preserves a selected peg across carrier relocation."""

import json

import gkm_try

from perception import safe_step
from probe_level9 import pieces, symbolic_solution


def compact(node):
    state = pieces(node.frame())
    return (
        state["pegs"], state["carriers"], state["selected"],
        state["bridges"], int(node.levels_completed),
    )


def probe(env):
    with open("checkpoint.json") as checkpoint_file:
        prefix = json.load(checkpoint_file)["final_path"]
    for action in prefix:
        safe_step(env, action)
    solution, _ = symbolic_solution(env.frame())
    loaded = env.clone()
    for _, source, destination in solution:
        safe_step(loaded, (6, source[1] + 1, source[0] + 1))
        safe_step(loaded, (6, destination[1] + 1, destination[0] + 1))

    for key in (1, 2, 3, 4):
        node = loaded.clone()
        safe_step(node, 7)
        safe_step(node, key)
        before = compact(node)
        state = pieces(node.frame())
        if state["carriers"]:
            row, col = state["carriers"][0]
            safe_step(node, (6, col + 1, row + 1))
        print("RESET_LOAD", key, before, compact(node))

    paths = (
        (7, 4, (6, 61, 37), 4),
        (7, 4, (6, 61, 37), 3),
        (7, 4, (6, 61, 37), 7),
        (7, 4, (6, 43, 37)),
        (7, 3, (6, 37, 37)),
    )
    for path in paths:
        node = loaded.clone()
        for action in path:
            safe_step(node, action)
        print("RESET_PATH", path, compact(node))


gkm_try.A.run_program("lf52", probe)

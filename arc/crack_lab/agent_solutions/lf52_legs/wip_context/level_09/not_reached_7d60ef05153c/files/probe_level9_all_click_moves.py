"""Enumerate every in-frame source/destination click effect on level 9."""

import json

import gkm_try

from perception import safe_step
from probe_level9 import pieces, symbolic_solution


def state_key(node):
    state = pieces(node.frame())
    return state["pegs"], state["bridges"], state["carriers"]


def enumerate_moves(root, label):
    state = pieces(root.frame())
    sources = tuple(("P", point) for point in state["pegs"])
    sources += tuple(("B", point) for point in state["bridges"])
    sources += tuple(("F", point) for point in state["fixed_bridges"])
    sources += tuple(("C", point) for point in state["carriers"])
    destinations = tuple(sorted(
        set(state["holes"]) | set(state["carriers"]) |
        set(state["pegs"]) | set(state["bridges"])
    ))
    before = state_key(root)
    effects = []
    for kind, source in sources:
        for destination in destinations:
            if destination == source:
                continue
            node = root.clone()
            safe_step(node, (6, source[1] + 1, source[0] + 1))
            safe_step(node, (6, destination[1] + 1, destination[0] + 1))
            after = state_key(node)
            if after != before or int(node.levels_completed) != int(root.levels_completed):
                effects.append((
                    kind, source, destination,
                    abs(destination[0] - source[0]) + abs(destination[1] - source[1]),
                    after[0], after[1], int(node.levels_completed),
                ))
    print("ALL_CLICKS", label, len(sources), len(destinations), len(effects), tuple(effects))


def probe(env):
    with open("checkpoint.json") as checkpoint_file:
        prefix = json.load(checkpoint_file)["final_path"]
    for action in prefix:
        safe_step(env, action)
    enumerate_moves(env, "entry")

    solution, _ = symbolic_solution(env.frame())
    loaded = env.clone()
    for _, source, destination in solution:
        safe_step(loaded, (6, source[1] + 1, source[0] + 1))
        safe_step(loaded, (6, destination[1] + 1, destination[0] + 1))
    for action in (4,) * 9:
        safe_step(loaded, action)
    enumerate_moves(loaded, "far_aligned")

    staged = loaded.clone()
    for action in ((6, 59, 19), (6, 47, 19), 3):
        safe_step(staged, action)
    enumerate_moves(staged, "far_staged")


gkm_try.A.run_program("lf52", probe)

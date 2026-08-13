"""Compact contextual action-7 probes on level 9."""

import json

import gkm_try

from perception import arr, safe_step
from probe_level9 import pieces, symbolic_solution


def signature(node):
    return int(node.levels_completed), pieces(node.frame()), arr(node.frame())[1:, :].tobytes()


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

    paths = (
        (), (7,), (4,), (7, 4), (4, 7), (7, 4, 4), (4, 7, 4),
        ((6, 59, 19),), ((6, 59, 19), 7),
        ((6, 59, 19), 7, (6, 47, 19)),
        (7, (6, 59, 19), (6, 47, 19)),
    )
    baseline = {}
    for path in paths:
        node = loaded.clone()
        for action in path:
            safe_step(node, action)
        level, state, raw = signature(node)
        baseline[path] = raw
        print("L9_ACTION7", path, level, state)
    comparisons = (
        ((7, 4), (4,)),
        ((4, 7), (4,)),
        ((7, 4, 4), (4, 4)),
        (((6, 59, 19), 7), ((6, 59, 19),)),
    )
    print("L9_ACTION7_EQUAL", tuple((left, right, baseline[left] == baseline[right]) for left, right in comparisons))


gkm_try.A.run_program("lf52", probe)

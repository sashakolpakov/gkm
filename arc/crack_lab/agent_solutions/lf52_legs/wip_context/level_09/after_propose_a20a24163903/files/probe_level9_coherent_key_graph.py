"""Enumerate left/right transitions after the two-cycle coherence reset."""

from collections import deque
import json

import gkm_try

from perception import arr, safe_step
from probe_level9 import pieces, symbolic_solution
from probe_level9_coherence_cycle import SOURCE, DESTINATION


def frame_key(node):
    return arr(node.frame())[1:, :].tobytes()


def compact(node):
    state = pieces(node.frame())
    return state["carriers"], state["pegs"], state["bridges"]


def probe(env):
    with open("checkpoint.json") as checkpoint_file:
        prefix = json.load(checkpoint_file)["final_path"]
    for action in prefix:
        safe_step(env, action)
    solution, _ = symbolic_solution(env.frame())
    for _, source, destination in solution:
        safe_step(env, (6, source[1] + 1, source[0] + 1))
        safe_step(env, (6, destination[1] + 1, destination[0] + 1))
    cycle = (7, 7, SOURCE, DESTINATION)
    for action in cycle * 2:
        safe_step(env, action)

    queue = deque([(env.clone(), ())]); seen = {frame_key(env)}; rows = []
    while queue and len(seen) <= 120:
        node, path = queue.popleft(); rows.append((path, compact(node)))
        if len(path) >= 12:
            continue
        for action in (3, 4):
            child = node.clone(); safe_step(child, action)
            key = frame_key(child)
            if key in seen:
                continue
            seen.add(key); queue.append((child, path + (action,)))
    print("COHERENT_KEY_GRAPH", len(seen), len(rows))
    for row in rows:
        print("COHERENT_KEY_STATE", row)


gkm_try.A.run_program("lf52", probe)

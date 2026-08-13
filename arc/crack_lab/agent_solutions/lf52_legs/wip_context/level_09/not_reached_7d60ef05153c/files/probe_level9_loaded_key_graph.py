"""Enumerate the contextual key graph of level 9's loaded carrier."""

from collections import deque
import json

import gkm_try

from perception import arr, safe_step
from probe_level9 import pieces, symbolic_solution


def frame_key(node):
    return arr(node.frame())[1:, :].tobytes()


def compact(node):
    state = pieces(node.frame())
    return state["carriers"], state["pegs"], state["bridges"], state["fixed_bridges"]


def probe(env):
    with open("checkpoint.json") as checkpoint_file:
        prefix = json.load(checkpoint_file)["final_path"]
    for action in prefix:
        safe_step(env, action)
    solution, _ = symbolic_solution(env.frame())
    for _, source, destination in solution:
        safe_step(env, (6, source[1] + 1, source[0] + 1))
        safe_step(env, (6, destination[1] + 1, destination[0] + 1))

    queue = deque([(env.clone(), ())])
    seen = {frame_key(env)}
    vertical = []
    states = []
    while queue and len(seen) <= 500:
        node, path = queue.popleft()
        states.append((path, compact(node)))
        if len(path) >= 24:
            continue
        before = frame_key(node)
        for action in (1, 2, 3, 4):
            child = node.clone(); safe_step(child, action)
            key = frame_key(child)
            if action in (1, 2) and key != before:
                vertical.append((path + (action,), compact(child)))
            if key in seen:
                continue
            seen.add(key); queue.append((child, path + (action,)))
    print("LOADED_KEY_GRAPH", len(seen), len(states), tuple(vertical))
    for path, state in states:
        print("LOADED_KEY_STATE", path, state)


gkm_try.A.run_program("lf52", probe)

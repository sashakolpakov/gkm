"""Bounded whole-level search over key actions and recognized peg macros."""

import json
import os
from heapq import heappop, heappush

import gkm_try
from legs import _bridge_carrier_moves, _bridge_carrier_state
from perception import arr


TARGET_LEVEL = int(os.environ.get("TARGET_LEVEL", "4"))
MAX_STATES = int(os.environ.get("MAX_STATES", "2000"))
MAX_COST = int(os.environ.get("MAX_COST", "50"))
LEVEL_ENDS = {1: 8, 2: 42, 3: 87, 4: 149, 5: 238, 6: 331, 7: 476, 8: 544}


def play(node, action):
    if isinstance(action, tuple):
        node.step(*action)
    else:
        node.step(action)


def solve(env):
    with open("checkpoint.json") as checkpoint_file:
        prefix = json.load(checkpoint_file)["final_path"][:LEVEL_ENDS[TARGET_LEVEL - 1]]
    for action in prefix:
        env.step(action)
    root = env.clone()

    def key(node):
        return node.levels_completed, arr(node.frame())[1:, :].tobytes()

    def dense(node):
        slots, pegs = _bridge_carrier_state(node.frame())[:2]
        return len(pegs), -len(slots)

    def estimate(node, cost):
        peg_count, negative_slots = dense(node)
        return cost + 7 * peg_count + negative_slots

    serial = 0
    root_key = key(root)
    queue = [(estimate(root, 0), 0, serial, root, (), root_key)]
    best_cost = {root_key: 0}
    best_dense = dense(root)
    best_path = ()
    solution = None

    while queue and len(best_cost) <= MAX_STATES:
        _, cost, _, node, path, node_key = heappop(queue)
        if cost != best_cost.get(node_key):
            continue
        if node.levels_completed >= TARGET_LEVEL:
            solution = path
            break
        if cost >= MAX_COST:
            continue

        macros = [(action,) for action in (1, 2, 3, 4)]
        macros += [
            (
                (6, source[1] + 1, source[0] + 1),
                (6, destination[1] + 1, destination[0] + 1),
            )
            for _, source, destination in _bridge_carrier_moves(node.frame())
        ]
        for macro in macros:
            child = node.clone()
            for action in macro:
                play(child, action)
            child_key = key(child)
            child_cost = cost + len(macro)
            if child_key == node_key or child_cost >= best_cost.get(child_key, 10 ** 9):
                continue
            child_path = path + macro
            best_cost[child_key] = child_cost
            child_dense = dense(child)
            if child_dense < best_dense:
                best_dense = child_dense
                best_path = child_path
            serial += 1
            heappush(
                queue,
                (estimate(child, child_cost), child_cost, serial, child, child_path, child_key),
            )

    print(
        "ASTAR_RESULT", TARGET_LEVEL, len(best_cost), best_dense,
        None if solution is None else len(solution),
    )
    print("ASTAR_PATH", solution)
    print("ASTAR_DENSE_PATH", best_path)


levels, path, error = gkm_try.A.run_program("lf52", solve)
print("ASTAR_RUN", levels, len(path), error)

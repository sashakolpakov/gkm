"""Find action-cost-optimal solutions for peg boards with carriers."""

import heapq
import json
import os

import gkm_try
from legs import _carrier_capture_macros
from perception import arr


TARGET_LEVEL = int(os.environ.get("TARGET_LEVEL", "3"))
MAX_STATES = int(os.environ.get("MAX_STATES", "8000"))
MAX_COST = int(os.environ.get("MAX_COST", "50"))
LEVEL_ENDS = {1: 8, 2: 42, 3: 87, 4: 149, 5: 238, 6: 331, 7: 476, 8: 544}


def key(node):
    return node.levels_completed, arr(node.frame())[1:, :].tobytes()


def probe(env):
    with open("checkpoint.json") as checkpoint_file:
        path = json.load(checkpoint_file)["final_path"]
    for action in path[:LEVEL_ENDS[TARGET_LEVEL - 1]]: env.step(action)
    base = env.levels_completed; root_key = key(env); distance = {root_key: 0}; serial = 0
    queue = [(0, serial, env.clone(), ())]; found = None
    while queue and len(distance) <= MAX_STATES:
        cost, _, node, action_path = heapq.heappop(queue)
        node_key = key(node)
        if cost != distance.get(node_key): continue
        if node.levels_completed > base:
            found = action_path; break
        if cost >= MAX_COST: continue
        macros = tuple((action,) for action in (1, 2, 3, 4))
        macros += _carrier_capture_macros(node.frame())
        for macro in macros:
            child = node.clone()
            for action in macro:
                if isinstance(action, tuple): child.step(*action)
                else: child.step(action)
            child_key = key(child); child_cost = cost + len(macro)
            if child_key == node_key or child_cost >= distance.get(child_key, 10 ** 9): continue
            distance[child_key] = child_cost; serial += 1
            heapq.heappush(queue, (child_cost, serial, child, action_path + macro))
    print("WEIGHTED_CARRIER", TARGET_LEVEL, len(distance), None if found is None else (len(found), found))


gkm_try.A.run_program("lf52", probe)

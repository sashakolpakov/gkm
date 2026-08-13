"""Action-weighted search for the early carrier peg-solitaire levels."""

import json
import os
import sys
from heapq import heappop, heappush

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from legs import _carrier_capture_macros
from perception import arr, safe_step


def normalize(action):
    return tuple(action) if isinstance(action, list) else action


def state_key(node):
    return int(node.levels_completed), arr(node.frame())[1:, :].tobytes()


def probe(env):
    desired = int(os.environ.get("OPT_LEVEL", "3"))
    max_states = int(os.environ.get("OPT_STATES", "2500"))
    bound = int(os.environ.get("OPT_COST", "45"))
    with open("checkpoint.json") as stream:
        campaign = tuple(normalize(action)
                         for action in json.load(stream)["final_path"])
    prior = int(env.levels_completed)
    entry = env.clone() if desired == 1 else None
    for action in campaign:
        safe_step(env, action)
        current = int(env.levels_completed)
        if prior < desired - 1 <= current:
            entry = env.clone()
            break
        prior = current

    base_level = int(entry.levels_completed)
    serial = 0
    queue = [(0, serial, entry.clone(), ())]
    best = {state_key(entry): 0}
    expanded = 0
    while queue and len(best) <= max_states:
        cost, _, node, path = heappop(queue)
        node_key = state_key(node)
        if cost != best.get(node_key) or cost >= bound:
            continue
        expanded += 1
        if expanded % 100 == 0:
            print("weighted_progress", desired, expanded, len(best), cost,
                  len(path), flush=True)
        macros = tuple((action,) for action in (1, 2, 3, 4))
        macros += _carrier_capture_macros(node.frame())
        for macro in macros:
            child_cost = cost + len(macro)
            if child_cost > bound:
                continue
            child = node.clone()
            for action in macro:
                safe_step(child, action)
                if int(child.levels_completed) > base_level:
                    solution = path + macro
                    print("weighted_solution", desired, len(solution),
                          len(best), expanded, solution, flush=True)
                    return
            child_key = state_key(child)
            if child_key == node_key or child_cost >= best.get(
                    child_key, 10 ** 9):
                continue
            best[child_key] = child_cost
            serial += 1
            heappush(queue, (child_cost, serial, child, path + macro))
    print("weighted_none", desired, bound, len(best), expanded, flush=True)


if __name__ == "__main__":
    levels, path, error = arena.run_program("lf52", probe)
    if error:
        print("weighted_worker_error", repr(error), flush=True)

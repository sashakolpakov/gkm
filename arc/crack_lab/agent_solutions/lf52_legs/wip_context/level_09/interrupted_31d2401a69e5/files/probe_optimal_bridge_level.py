"""Find a lowest-action solution for bridge-carrier levels 4 or 5."""

import json
import os
import sys
from heapq import heappop, heappush

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from legs import _bridge_carrier_moves, _bridge_carrier_state
from perception import safe_step


BOUNDARIES = (0, 8, 42, 87, 149, 238, 331, 476, 544)
ADMITTED_COST = {4: 62, 5: 89}


def state_key(node):
    return int(node.levels_completed), _bridge_carrier_state(node.frame())


def move_actions(move):
    _, source, destination = move
    return (
        (6, source[1] + 1, source[0] + 1),
        (6, destination[1] + 1, destination[0] + 1),
    )


def observe(env):
    level = int(os.environ.get("TARGET_LEVEL", "5"))
    if level not in (4, 5):
        raise ValueError("TARGET_LEVEL must be 4 or 5")
    with open("checkpoint.json") as stream:
        admitted = json.load(stream)["final_path"]
    prefix = BOUNDARIES[level - 1]
    for action in admitted[:prefix]:
        env.step(action)

    root = env.clone()
    base_level = int(root.levels_completed)
    serial = 0
    queue = [(0, serial, root, ())]
    best = {state_key(root): 0}
    expanded = 0
    solution = None
    max_cost = ADMITTED_COST[level] - 1
    max_states = 50000

    while queue and len(best) <= max_states:
        cost, _, node, path = heappop(queue)
        if best.get(state_key(node)) != cost:
            continue
        if node.levels_completed > base_level:
            solution = path
            break
        expanded += 1

        for action in (1, 2, 3, 4):
            child_cost = cost + 1
            if child_cost > max_cost:
                continue
            child = node.clone()
            safe_step(child, action)
            key = state_key(child)
            if child_cost < best.get(key, max_cost + 1):
                best[key] = child_cost
                serial += 1
                heappush(queue, (child_cost, serial, child,
                                 path + (action,)))

        if cost + 2 > max_cost:
            continue
        for move in _bridge_carrier_moves(node.frame()):
            child = node.clone()
            actions = move_actions(move)
            for action in actions:
                safe_step(child, action)
            child_cost = cost + 2
            key = state_key(child)
            if child_cost < best.get(key, max_cost + 1):
                best[key] = child_cost
                serial += 1
                child_path = path + actions
                if child.levels_completed > base_level:
                    solution = child_path
                    queue.clear()
                    break
                heappush(queue, (child_cost, serial, child, child_path))

    print("SEARCH", {"level": level, "states": len(best),
                     "expanded": expanded, "remaining": len(queue),
                     "solution_cost": None if solution is None else len(solution)})
    print("SOLUTION", solution)


levels, path, error = arena.run_program("lf52", observe)
print("PROBE_RESULT", {"levels": levels, "moves": len(path), "error": str(error)})

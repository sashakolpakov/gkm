"""Progress-first bounded search from level 5's reproduced second board."""

import heapq
import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from legs import _bridge_carrier_moves, _bridge_carrier_state
from perception import safe_step


PREFIX = 183


def move_actions(move):
    _, source, destination = move
    return ((6, source[1] + 1, source[0] + 1),
            (6, destination[1] + 1, destination[0] + 1))


def observe(env):
    with open("checkpoint.json") as stream:
        path = json.load(stream)["final_path"]
    for action in path[:PREFIX]:
        safe_step(env, action)
    root = env.clone()
    base_level = int(root.levels_completed)
    root_state = _bridge_carrier_state(root.frame())
    limit = int(os.environ.get("STATE_LIMIT", "10000"))
    cost_limit = int(os.environ.get("COST_LIMIT", "54"))

    nodes = [(root, -1, None, 0, root_state)]
    best = {root_state: 0}
    queue = [(len(root_state[1]), 0, 0, 0)]
    serial = 0
    goal = None
    expanded = 0
    while queue and len(nodes) < limit:
        _, cost, _, index = heapq.heappop(queue)
        node, _, _, node_cost, state = nodes[index]
        if node_cost != cost or best.get(state) != cost:
            continue
        expanded += 1
        transitions = [((action,), 1) for action in (1, 2, 3, 4)]
        transitions += [(move_actions(move), 2)
                        for move in _bridge_carrier_moves(node.frame())]
        for actions, edge_cost in transitions:
            child_cost = cost + edge_cost
            if child_cost > cost_limit:
                continue
            child = node.clone()
            for action in actions:
                safe_step(child, action)
            if child.levels_completed > base_level:
                child_state = _bridge_carrier_state(child.frame())
                nodes.append((child, index, actions, child_cost, child_state))
                goal = len(nodes) - 1
                queue.clear()
                break
            child_state = _bridge_carrier_state(child.frame())
            if child_cost >= best.get(child_state, 10 ** 9):
                continue
            best[child_state] = child_cost
            nodes.append((child, index, actions, child_cost, child_state))
            serial += 1
            heapq.heappush(queue, (len(child_state[1]), child_cost,
                                   serial, len(nodes) - 1))
        if expanded % 1000 == 0:
            print("BEAM_PROGRESS", {"expanded": expanded,
                                    "states": len(best),
                                    "queue": len(queue), "cost": cost,
                                    "pegs": len(state[1])})

    if goal is None:
        print("BEAM_RESULT", {"goal": None, "expanded": expanded,
                              "states": len(best), "queue": len(queue)})
        return
    chunks = []
    index = goal
    while nodes[index][1] >= 0:
        chunks.append(nodes[index][2])
        index = nodes[index][1]
    solution = tuple(action for chunk in reversed(chunks) for action in chunk)
    print("BEAM_RESULT", {"goal": True, "cost": len(solution),
                          "expanded": expanded, "states": len(best),
                          "path": solution})


levels, path, error = arena.run_program("lf52", observe)
print("PROBE_RESULT", {"levels": levels, "moves": len(path),
                       "error": str(error)})

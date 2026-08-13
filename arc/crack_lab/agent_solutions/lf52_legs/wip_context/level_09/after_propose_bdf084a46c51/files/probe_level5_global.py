"""Bounded best-first search for a shorter complete level-5 route."""

import heapq
import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from legs import _bridge_carrier_moves, _bridge_carrier_state
from perception import safe_step


PREFIX = 149


def frame_key(env):
    return env.frame().tobytes()


def progress(frame):
    state = _bridge_carrier_state(frame)
    # Merely an ordering hint: fewer visible pegs first, then more revealed
    # geometry.  Cost remains the first priority within a small slack band.
    return len(state[1]), -(len(state[0]) + len(state[2]) + len(state[3]))


def observe(env):
    with open("checkpoint.json") as stream:
        prefix = json.load(stream)["final_path"][:PREFIX]
    for action in prefix:
        safe_step(env, action)

    root = env.clone()
    base_level = root.levels_completed
    limit = int(os.environ.get("STATE_LIMIT", "50000"))
    incumbent = int(os.environ.get("COST_LIMIT", "89"))
    slack = int(os.environ.get("COST_SLACK", "10"))

    nodes = [(root, -1, None, 0)]
    root_key = frame_key(root)
    best = {root_key: 0}
    serial = 0
    queue = []
    peg_hint, geometry_hint = progress(root.frame())
    heapq.heappush(queue, (0, peg_hint, geometry_hint, serial, 0))
    goal_index = None

    while queue and len(nodes) < limit:
        _, _, _, _, index = heapq.heappop(queue)
        node, _, _, cost = nodes[index]
        key = frame_key(node)
        if best.get(key) != cost:
            continue
        if cost >= incumbent:
            continue

        transitions = [((action,), 1) for action in (1, 2, 3, 4)]
        transitions.extend(
            (((6, source[1] + 1, source[0] + 1),
              (6, destination[1] + 1, destination[0] + 1)), 2)
            for _, source, destination in _bridge_carrier_moves(node.frame())
        )
        for actions, action_cost in transitions:
            child_cost = cost + action_cost
            if child_cost >= incumbent:
                continue
            child = node.clone()
            for action in actions:
                safe_step(child, action)
            if child.levels_completed > base_level:
                nodes.append((child, index, actions, child_cost))
                goal_index = len(nodes) - 1
                incumbent = child_cost
                print("GOAL", {"cost": child_cost, "states": len(nodes)})
                queue.clear()
                break
            child_key = frame_key(child)
            if child_cost >= best.get(child_key, incumbent):
                continue
            best[child_key] = child_cost
            nodes.append((child, index, actions, child_cost))
            child_index = len(nodes) - 1
            serial += 1
            peg_hint, geometry_hint = progress(child.frame())
            # Mostly uniform-cost, with bounded progress bias among nodes no
            # more than `slack` actions beyond the cheapest frontier.
            priority = max(0, child_cost - slack)
            heapq.heappush(
                queue,
                (priority, peg_hint, geometry_hint, serial, child_index),
            )
        if len(nodes) % 5000 < 8:
            print("SEARCH", {"states": len(nodes), "queue": len(queue),
                             "cost": cost, "best_cost": incumbent})

    if goal_index is None:
        print("NO_GOAL", {"states": len(nodes), "queue": len(queue),
                          "seen": len(best), "cost_limit": incumbent})
        return

    chunks = []
    index = goal_index
    while nodes[index][1] >= 0:
        chunks.append(nodes[index][2])
        index = nodes[index][1]
    path = [action for chunk in reversed(chunks) for action in chunk]
    print("GLOBAL_RESULT", {"cost": len(path), "path": path})


levels, path, error = arena.run_program("lf52", observe)
print("PROBE_RESULT", {"levels": levels, "moves": len(path),
                       "error": str(error)})

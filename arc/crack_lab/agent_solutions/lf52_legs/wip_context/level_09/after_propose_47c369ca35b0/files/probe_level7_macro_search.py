"""Cost-aware search over level-7 cargo moves and carrier-key orbits."""

from collections import deque
import heapq
import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from probe_level7_carrier_opportunities import (
    move_actions,
    physical_key,
    play,
    summary,
    valid_move,
    visible_moves,
)


MAX_COST = int(os.environ.get("LEVEL7_MAX_COST", "110"))
MAX_MACROS = int(os.environ.get("LEVEL7_MAX_MACROS", "1000"))
CARRIER_DEPTH = int(os.environ.get("CARRIER_MAX_DEPTH", "18"))
CARRIER_STATES = int(os.environ.get("CARRIER_MAX_STATES", "200"))


def carrier_graph(root):
    def reconstruct(path):
        node = root.clone()
        for action in path:
            node.step(action)
        return node

    queue = deque([()])
    seen = {physical_key(root)}
    opportunities = {}
    while queue and len(seen) <= CARRIER_STATES:
        key_path = queue.popleft()
        node = reconstruct(key_path)
        for move in visible_moves(node.frame()):
            child = node.clone()
            actions = move_actions(move)
            for action in actions:
                play(child, action)
            if not valid_move(node, child, move):
                continue
            child_key = physical_key(child)
            edge = (key_path + tuple(actions), child, move)
            previous = opportunities.get(child_key)
            if previous is None or len(edge[0]) < len(previous[0]):
                opportunities[child_key] = edge
        if len(key_path) >= CARRIER_DEPTH:
            continue
        for action in (1, 2, 3, 4):
            child_path = key_path + (action,)
            child = reconstruct(child_path)
            key = physical_key(child)
            if key in seen:
                continue
            seen.add(key)
            queue.append(child_path)
    orbit = tuple(sorted(hash(key) for key in seen))
    return orbit, tuple(opportunities.values())


def search(root):
    serial = 0
    queue = [(0, serial, root.clone(), [])]
    best = {}
    expanded = 0
    while queue and expanded < MAX_MACROS:
        cost, _, node, path = heapq.heappop(queue)
        if cost > MAX_COST:
            break
        orbit, edges = carrier_graph(node)
        state_key = (physical_key(node), orbit)
        if cost >= best.get(state_key, MAX_COST + 1):
            continue
        best[state_key] = cost
        expanded += 1
        print("STATE", {
            "expanded": expanded,
            "cost": cost,
            "carrier_states": len(orbit),
            "opportunities": len(edges),
            "summary": summary(node),
        }, flush=True)
        for edge_actions, child, move in edges:
            child_cost = cost + len(edge_actions)
            if child_cost > MAX_COST:
                continue
            child_path = path + list(edge_actions)
            if child.levels_completed > 6:
                return child_path, expanded, len(best), move
            serial += 1
            heapq.heappush(
                queue,
                (child_cost, serial, child, child_path),
            )
    return None, expanded, len(best), None


def probe(env):
    with open("checkpoint.json") as checkpoint_file:
        campaign = json.load(checkpoint_file)["final_path"]
    for action in campaign[:331]:
        play(env, action)
    root = env.clone()
    result = search(root)
    print("SEARCH", {
        "result": result,
        "cost": len(result[0]) if result[0] else None,
    }, flush=True)


levels, path, error = arena.run_program("lf52", probe)
print("HARNESS", {"levels": levels, "moves": len(path), "error": str(error) if error else None})

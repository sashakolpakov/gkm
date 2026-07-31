"""Action-cost Dijkstra for the persistent bridge/carrier level."""

import heapq
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from legs import _bridge_carrier_moves, _bridge_carrier_state


MAX_COST = int(os.environ.get("LEVEL5_MAX_COST", "60"))
MAX_STATES = int(os.environ.get("LEVEL5_MAX_STATES", "5000"))


def play(env, action):
    if isinstance(action, list):
        env.step(*action)
    else:
        env.step(action)


def physical_key(env):
    frame = np.asarray(env.frame()).copy()
    frame[0, :] = 0
    return frame.tobytes()


def move_actions(move):
    _, source, destination = move
    return (
        [6, source[1] + 1, source[0] + 1],
        [6, destination[1] + 1, destination[0] + 1],
    )


def dense(env):
    slots, pegs, carriers, bridges, _, _ = _bridge_carrier_state(env.frame())
    distances = [
        abs(first[0] - second[0]) + abs(first[1] - second[1])
        for index, first in enumerate(sorted(pegs))
        for second in sorted(pegs)[index + 1:]
    ]
    return (
        len(pegs),
        min(distances, default=999),
        -len(slots),
        len(carriers),
        len(bridges),
    )


def search(root):
    serial = 0
    queue = [(0, serial, root.clone(), [])]
    best = {physical_key(root): 0}
    expanded = 0
    best_dense = (dense(root), 0, [])
    while queue and expanded < MAX_STATES:
        cost, _, node, path = heapq.heappop(queue)
        if cost > MAX_COST:
            break
        key = physical_key(node)
        if cost != best.get(key):
            continue
        expanded += 1
        score = dense(node)
        if score < best_dense[0]:
            best_dense = (score, cost, path)
            print("DENSE", {
                "expanded": expanded,
                "cost": cost,
                "score": score,
                "tail": path[-10:],
            }, flush=True)
        children = []
        for action in (1, 2, 3, 4):
            child = node.clone()
            child.step(action)
            if physical_key(child) != key:
                children.append((1, child, [action]))
        for move in _bridge_carrier_moves(node.frame()):
            child = node.clone()
            actions = list(move_actions(move))
            for action in actions:
                play(child, action)
            if child.levels_completed > 4:
                return path + actions, expanded, len(best), best_dense
            if physical_key(child) != key:
                children.append((2, child, actions))
        for edge_cost, child, actions in children:
            child_cost = cost + edge_cost
            if child_cost > MAX_COST:
                continue
            child_key = physical_key(child)
            if child_cost >= best.get(child_key, MAX_COST + 1):
                continue
            best[child_key] = child_cost
            serial += 1
            heapq.heappush(
                queue,
                (child_cost, serial, child, path + actions),
            )
        if expanded % 250 == 0:
            print("PROGRESS", {
                "expanded": expanded,
                "seen": len(best),
                "cost": cost,
            }, flush=True)
    return None, expanded, len(best), best_dense


def probe(env):
    with open("checkpoint.json") as checkpoint_file:
        campaign = json.load(checkpoint_file)["final_path"]
    for action in campaign[:149]:
        play(env, action)
    result = search(env.clone())
    print("SEARCH", {
        "result": result,
        "cost": len(result[0]) if result[0] else None,
    }, flush=True)


levels, path, error = arena.run_program("lf52", probe)
print("HARNESS", {"levels": levels, "moves": len(path), "error": str(error) if error else None})

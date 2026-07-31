"""Weighted macro search for level 5 over carrier alignments and peg moves."""

from collections import deque
import heapq
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from legs import _bridge_carrier_moves, _bridge_carrier_state


MAX_COST = int(os.environ.get("LEVEL5_MAX_COST", "60"))
MAX_MACROS = int(os.environ.get("LEVEL5_MAX_MACROS", "300"))
CARRIER_DEPTH = int(os.environ.get("CARRIER_MAX_DEPTH", "16"))
CARRIER_STATES = int(os.environ.get("CARRIER_MAX_STATES", "650"))


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


def summary(env):
    slots, pegs, carriers, bridges, borders, selected = (
        _bridge_carrier_state(env.frame())
    )
    return {
        "slots": len(slots),
        "pegs": sorted(pegs),
        "carriers": sorted(carriers),
        "bridges": sorted(bridges),
        "borders": sorted(borders),
        "selected": selected,
        "level": env.levels_completed,
    }


def carrier_graph(root):
    node = root.clone()
    seen_depth = {physical_key(node): 0}
    opportunities = {}
    inverses = {1: 2, 2: 1, 3: 4, 4: 3}

    def visit(key_path):
        nonlocal node
        if len(seen_depth) > CARRIER_STATES:
            return
        node_key = physical_key(node)
        for move in _bridge_carrier_moves(node.frame()):
            actions = move_actions(move)
            child = root.clone()
            for action in key_path:
                play(child, action)
            before = physical_key(child)
            for action in actions:
                play(child, action)
            if child.levels_completed <= 4 and physical_key(child) == before:
                continue
            child_key = physical_key(child)
            edge = (key_path + tuple(actions), child, move)
            previous = opportunities.get(child_key)
            if previous is None or len(edge[0]) < len(previous[0]):
                opportunities[child_key] = edge
        if len(key_path) >= CARRIER_DEPTH:
            return
        for action in (1, 2, 3, 4):
            play(node, action)
            child_key = physical_key(node)
            if child_key == node_key:
                continue
            child_depth = len(key_path) + 1
            if child_depth < seen_depth.get(child_key, CARRIER_DEPTH + 1):
                seen_depth[child_key] = child_depth
                visit(key_path + (action,))
            play(node, inverses[action])
            if physical_key(node) != node_key:
                node = root.clone()
                for restore_action in key_path:
                    play(node, restore_action)
                if physical_key(node) != node_key:
                    raise RuntimeError(
                        "carrier path failed to reconstruct state"
                    )

    visit(())
    orbit = tuple(sorted(hash(key) for key in seen_depth))
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
        key = (physical_key(node), orbit)
        if cost >= best.get(key, MAX_COST + 1):
            continue
        best[key] = cost
        expanded += 1
        print("STATE", {
            "expanded": expanded,
            "cost": cost,
            "carrier_states": len(orbit),
            "opportunities": len(edges),
            "summary": summary(node),
        }, flush=True)
        for actions, child, move in edges:
            child_cost = cost + len(actions)
            if child_cost > MAX_COST:
                continue
            child_path = path + list(actions)
            if child.levels_completed > 4:
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
    for action in campaign[:149]:
        play(env, action)
    result = search(env.clone())
    print("SEARCH", {
        "result": result,
        "cost": len(result[0]) if result[0] else None,
    }, flush=True)


levels, path, error = arena.run_program("lf52", probe)
print("HARNESS", {"levels": levels, "moves": len(path), "error": str(error) if error else None})

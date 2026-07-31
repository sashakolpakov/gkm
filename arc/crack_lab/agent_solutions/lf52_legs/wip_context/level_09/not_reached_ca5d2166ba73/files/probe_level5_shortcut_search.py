"""Find cheaper local rejoins in the verified level-5 macro route."""

import heapq
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from legs import _bridge_carrier_moves


CONTEXT_INDEX = int(os.environ.get("CONTEXT_INDEX", "34"))
MAX_COST = int(os.environ.get("SHORTCUT_MAX_COST", "35"))
MAX_STATES = int(os.environ.get("SHORTCUT_MAX_STATES", "500"))


def play(env, action):
    if isinstance(action, list):
        env.step(*action)
    else:
        env.step(action)


def physical_key(env):
    frame = np.asarray(env.frame()).copy()
    frame[0, :] = 0
    return frame.tobytes()


def frame_key(env):
    frame = np.asarray(env.frame()).copy()
    frame[0, :] = 0
    return frame


def move_actions(move):
    _, source, destination = move
    return (
        [6, source[1] + 1, source[0] + 1],
        [6, destination[1] + 1, destination[0] + 1],
    )


def search(root, targets, target_frames):
    def priority(node, cost):
        frame = frame_key(node)
        distances = [
            int(np.count_nonzero(frame != target_frame))
            for index, target_frame in target_frames
            if index - CONTEXT_INDEX > cost
        ]
        return min(distances, default=4096)

    serial = 0
    start = root.clone()
    queue = [(priority(start, 0), 0, serial, [], start)]
    best = {physical_key(start): 0}
    expanded = 0
    while queue and expanded < MAX_STATES:
        _, cost, _, path, node = heapq.heappop(queue)
        key = physical_key(node)
        if cost != best.get(key):
            continue
        target_index = targets.get(key)
        if target_index is not None and target_index - CONTEXT_INDEX > cost:
            return path, target_index, expanded, len(best)
        expanded += 1
        children = []
        if cost + 1 <= MAX_COST:
            for action in (1, 2, 3, 4):
                child = node.clone()
                play(child, action)
                if physical_key(child) != key:
                    children.append((1, child, [action]))
        if cost + 2 <= MAX_COST:
            for move in _bridge_carrier_moves(node.frame()):
                actions = list(move_actions(move))
                child = node.clone()
                for action in actions:
                    play(child, action)
                if child.levels_completed > 4 or physical_key(child) != key:
                    children.append((2, child, actions))
        for edge_cost, child, actions in children:
            child_cost = cost + edge_cost
            child_key = physical_key(child)
            if child_cost >= best.get(child_key, MAX_COST + 1):
                continue
            best[child_key] = child_cost
            serial += 1
            heapq.heappush(
                queue,
                (
                    priority(child, child_cost), child_cost, serial,
                    path + actions, child,
                ),
            )
        if expanded % 100 == 0:
            print("PROGRESS", {
                "expanded": expanded,
                "seen": len(best),
                "cost": cost,
            }, flush=True)
    return None, None, expanded, len(best)


def probe(env):
    with open("campaign_candidate_633.json") as campaign_file:
        campaign = json.load(campaign_file)
    with open("level5_ddmin_89.json") as candidate_file:
        candidate = json.load(candidate_file)
    for action in campaign[:137]:
        play(env, action)
    root = env.clone()
    for action in candidate[:CONTEXT_INDEX]:
        play(root, action)

    target_node = root.clone()
    targets = {}
    target_frames = []
    for index, action in enumerate(
        candidate[CONTEXT_INDEX:], CONTEXT_INDEX + 1
    ):
        play(target_node, action)
        targets[physical_key(target_node)] = index
        target_frames.append((index, frame_key(target_node)))
    path, target_index, expanded, seen = search(
        root, targets, target_frames
    )
    print("SHORTCUT", {
        "context": CONTEXT_INDEX,
        "path": path,
        "cost": len(path) if path else None,
        "target": target_index,
        "replaced": (
            target_index - CONTEXT_INDEX if target_index is not None else None
        ),
        "saving": (
            target_index - CONTEXT_INDEX - len(path)
            if target_index is not None else None
        ),
        "expanded": expanded,
        "seen": seen,
    }, flush=True)


if __name__ == "__main__":
    levels, path, error = arena.run_program("lf52", probe)
    print("HARNESS", {
        "levels": levels,
        "moves": len(path),
        "error": str(error) if error else None,
    })

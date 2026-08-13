"""Find a cheaper local path between verified level-9 route checkpoints."""

import heapq
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from probe_level9_shortest_suffix import (
    move_actions,
    physical_key,
    play,
    visible_moves,
)


CONTEXT_INDEX = int(os.environ.get("CONTEXT_INDEX", "62"))
MAX_COST = int(os.environ.get("SHORTCUT_MAX_COST", "15"))
MAX_STATES = int(os.environ.get("SHORTCUT_MAX_STATES", "1200"))
GREEDY = os.environ.get("SHORTCUT_GREEDY") == "1"


def move_succeeded(node, move):
    if node.levels_completed > 8:
        return True
    kind, source, destination = move
    color = 9 if kind == "bridge" else 14
    array = np.asarray(node.frame())

    def count(position):
        row, col = position
        return int(np.count_nonzero(
            array[row:row + 4, col:col + 4] == color
        ))

    return count(source) < 12 and count(destination) >= 12


def search(root, targets, target_frames):
    def priority(node, cost):
        if not GREEDY:
            return cost
        frame = np.asarray(node.frame()).copy()
        frame[0, :] = 0
        distances = [
            int(np.count_nonzero(frame != target_frame))
            for index, target_frame in target_frames
            if index - CONTEXT_INDEX > cost
        ]
        return min(distances, default=4096)

    serial = 0
    start = root.clone()
    queue = [(priority(start, 0), 0, serial, [], start, None)]
    best = {physical_key(root): 0}
    expanded = 0
    while queue and expanded < MAX_STATES:
        _, cost, _, path, node, forbidden = heapq.heappop(queue)
        if cost > MAX_COST:
            break
        key = physical_key(node)
        if cost != best.get(key):
            continue
        target_index = targets.get(key)
        if target_index is not None and target_index - CONTEXT_INDEX > cost:
            return path, target_index, expanded, len(best)
        expanded += 1
        children = []
        if cost + 1 <= MAX_COST:
            for action in (3, 4):
                if forbidden == ("key", action):
                    continue
                child = node.clone()
                play(child, action)
                if physical_key(child) != key:
                    inverse = ("key", 4 if action == 3 else 3)
                    children.append((1, child, [action], inverse))
        if cost + 2 <= MAX_COST:
            for move in visible_moves(node.frame()):
                if forbidden == ("move", move[0], move[1], move[2]):
                    continue
                actions = list(move_actions(move[1], move[2]))
                child = node.clone()
                for action in actions:
                    play(child, action)
                if move_succeeded(child, move):
                    inverse = ("move", move[0], move[2], move[1])
                    children.append((2, child, actions, inverse))
        for edge_cost, child, actions, inverse in children:
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
                (
                    priority(child, child_cost), child_cost, serial,
                    path + actions, child, inverse,
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
    with open("checkpoint.json") as checkpoint_file:
        prefix = json.load(checkpoint_file)["final_path"]
    with open("level9_candidate_102.json") as candidate_file:
        candidate = json.load(candidate_file)
    for action in prefix:
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
        frame = np.asarray(target_node.frame()).copy()
        frame[0, :] = 0
        target_frames.append((index, frame))
    result = search(root, targets, target_frames)
    path, target_index, expanded, seen = result
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

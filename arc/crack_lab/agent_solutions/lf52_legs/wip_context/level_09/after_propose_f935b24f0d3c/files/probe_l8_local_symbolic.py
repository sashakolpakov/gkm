"""Solve level 8's first two-bridge board exactly after carrier alignment."""

from collections import deque
import json
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from legs import _bridge_carrier_state, _movable_bridge_board
from perception import safe_step


ENTRY = 476
ALIGNMENT = (3, 3, 1, 1, 1, 1, 4)
WRAP_KEYS = (3, 2, 2, 2, 2, 4, 4, 4, 2, 3, 3, 3)
DIRECTIONS = ((-6, 0), (6, 0), (0, -6), (0, 6))


def play(node, action):
    safe_step(node, tuple(action) if isinstance(action, list) else action)


def extract(frame):
    slots, carriers, bridges, pegs = _movable_bridge_board(frame)
    array = np.asarray(frame)
    windows = np.lib.stride_tricks.sliding_window_view(array, (4, 4))
    counts = {
        color: np.count_nonzero(windows == color, axis=(-1, -2))
        for color in (1, 9, 12, 14)
    }

    def positions(mask):
        rows, cols = np.where(mask)
        return set(zip(map(int, rows), map(int, cols)))

    slots |= positions(counts[1] == 16)
    carriers |= positions(counts[12] == 16)
    bridges |= positions(counts[9] >= 12)
    pegs |= positions(counts[14] >= 12)
    fixed = _bridge_carrier_state(frame)
    slots |= set(fixed[0])
    carriers |= set(fixed[2])
    pegs |= set(fixed[1])
    return (frozenset(slots), frozenset(carriers), frozenset(bridges),
            frozenset(pegs), frozenset(fixed[3]))


def solve(frame, target=(36, 12), extra_goals=4):
    slots, carriers, bridges, pegs, fixed = extract(frame)
    start = (bridges, pegs)
    queue = deque([(start, ())])
    seen = {start}
    destinations = slots | carriers
    goals = []
    min_goal = None
    while queue:
        (state_bridges, state_pegs), path = queue.popleft()
        reached = (bool(state_pegs & carriers) if target is None
                   else target in state_pegs)
        if reached:
            if min_goal is None:
                min_goal = len(path)
            if len(path) <= min_goal + extra_goals:
                goals.append((path, state_bridges, state_pegs))
            continue
        if min_goal is not None and len(path) >= min_goal + extra_goals:
            continue
        occupied = state_bridges | state_pegs | fixed
        for kind, pieces in (("bridge", state_bridges), ("peg", state_pegs)):
            for source in sorted(pieces):
                for dr, dc in DIRECTIONS:
                    midpoint = (source[0] + dr, source[1] + dc)
                    destination = (source[0] + 2 * dr, source[1] + 2 * dc)
                    if (midpoint not in occupied
                            or destination not in destinations
                            or destination in occupied):
                        continue
                    child_bridges = set(state_bridges)
                    child_pegs = set(state_pegs)
                    if kind == "bridge":
                        child_bridges.remove(source)
                        child_bridges.add(destination)
                    else:
                        child_pegs.remove(source)
                        child_pegs.add(destination)
                        child_pegs.discard(midpoint)
                    child = (frozenset(child_bridges), frozenset(child_pegs))
                    if child in seen:
                        continue
                    seen.add(child)
                    queue.append((child, path + ((kind, source, destination),)))
    if not goals:
        return None, len(seen), None, ()
    path, state_bridges, state_pegs = goals[0]
    return (path, len(seen),
            (slots, carriers, state_bridges, state_pegs, fixed), goals)


def alignment_worlds(root):
    def frame_key(node):
        array = np.asarray(node.frame()).copy()
        array[0, :] = 0
        return array.tobytes()

    queue = deque([(root.clone(), ())])
    seen = {frame_key(root)}
    worlds = {}
    while queue and len(seen) <= 80:
        node, path = queue.popleft()
        carriers = extract(node.frame())[1]
        previous = worlds.get(carriers)
        if previous is None or len(path) < len(previous[0]):
            worlds[carriers] = (path, node.clone())
        if len(path) >= 12:
            continue
        for action in (1, 2, 3, 4):
            child = node.clone()
            play(child, action)
            key = frame_key(child)
            if key in seen:
                continue
            seen.add(key)
            queue.append((child, path + (action,)))
    return worlds, len(seen)


def probe(env):
    with open("checkpoint.json") as stream:
        campaign = json.load(stream)["final_path"]
    for action in campaign[:ENTRY]:
        play(env, action)
    entry = env.clone()
    alignment_nodes, alignment_states = alignment_worlds(entry)
    alignment_results = []
    for carriers, (keys, node) in alignment_nodes.items():
        local_path, local_states, local_goal, _ = solve(
            node.frame(), target=None, extra_goals=0
        )
        if local_path is None:
            continue
        alignment_results.append({
            "keys": keys, "carriers": sorted(carriers),
            "local_moves": len(local_path),
            "cost": len(keys) + 2 * len(local_path),
            "local_path": local_path, "local_goal": local_goal,
            "local_states": local_states,
        })
    for action in ALIGNMENT:
        play(env, action)
    path, states, goal, goals = solve(env.frame())
    node = env.clone()
    for _, source, destination in path or ():
        play(node, (6, source[1] + 1, source[0] + 1))
        play(node, (6, destination[1] + 1, destination[0] + 1))
    wrapped = []
    for local_path, _, _ in goals:
        branch = env.clone()
        for _, source, destination in local_path:
            play(branch, (6, source[1] + 1, source[0] + 1))
            play(branch, (6, destination[1] + 1, destination[0] + 1))
        for action in WRAP_KEYS:
            play(branch, action)
        stage_path, stage_states, _, _ = solve(
            branch.frame(), target=(48, 24), extra_goals=0
        )
        wrapped.append({
            "local_moves": len(local_path),
            "world": extract(branch.frame()),
            "stage_moves": None if stage_path is None else len(stage_path),
            "stage_path": stage_path,
            "stage_states": stage_states,
        })
    print("L8_LOCAL", {"start": extract(env.frame()), "states": states,
                       "alignment_states": alignment_states,
                       "alignments": sorted(alignment_results,
                                            key=lambda item: item["cost"]),
                       "moves": None if path is None else len(path),
                       "path": path, "goal": goal,
                       "goal_depths": [len(item[0]) for item in goals],
                       "goal_bridges": [sorted(item[1]) for item in goals],
                       "goal_paths": [item[0] for item in goals],
                       "wrapped": wrapped,
                       "verified": extract(node.frame())}, flush=True)


levels, path, error = arena.run_program("lf52", probe)
print("HARNESS", {"levels": levels, "moves": len(path), "error": str(error)})

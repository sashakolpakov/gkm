"""Enumerate shortest level-9 entry solutions and test their wrapped worlds."""

from collections import Counter, deque
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena


MAX_DEPTH = int(os.environ.get("ENTRY_MAX_DEPTH", "18"))
MAX_GOALS = int(os.environ.get("ENTRY_MAX_GOALS", "500"))


def extract(frame):
    array = np.asarray(frame)
    destinations = set()
    bridges = set()
    pegs = set()
    carriers = set()
    for row in range(0, 61, 6):
        for col in range(0, 61, 6):
            patch = array[row:row + 4, col:col + 4]
            if int(np.isin(patch, (1, 9, 12, 14)).sum()) < 12:
                continue
            position = (row, col)
            destinations.add(position)
            if int(np.count_nonzero(patch == 14)) >= 12:
                pegs.add(position)
            elif int(np.count_nonzero(patch == 9)) >= 12:
                bridges.add(position)
            elif int(np.count_nonzero(patch == 12)) == 16:
                carriers.add(position)
    return frozenset(destinations), frozenset(bridges), frozenset(pegs), frozenset(carriers)


def goal_paths(frame):
    destinations, bridges, pegs, carriers = extract(frame)
    start = (bridges, pegs)
    queue = deque([(start, ())])
    seen = {start}
    goals = []
    while queue and len(goals) < MAX_GOALS:
        (state_bridges, state_pegs), path = queue.popleft()
        if len(state_pegs) == 1 and state_pegs <= carriers:
            goals.append(path)
            continue
        if len(path) >= MAX_DEPTH:
            continue
        occupied = state_bridges | state_pegs
        for kind, pieces in (("bridge", state_bridges), ("peg", state_pegs)):
            for source in sorted(pieces):
                for dr, dc in ((-6, 0), (6, 0), (0, -6), (0, 6)):
                    midpoint = (source[0] + dr, source[1] + dc)
                    destination = (source[0] + 2 * dr, source[1] + 2 * dc)
                    if (
                        midpoint not in occupied
                        or destination not in destinations
                        or destination in occupied
                    ):
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
                    queue.append((
                        child,
                        path + ((kind, source, destination),),
                    ))
    return goals, len(seen)


def play(env, action):
    if isinstance(action, list):
        env.step(*action)
    else:
        env.step(action)


def play_entry_path(env, path):
    actions = []
    for _, source, destination in path:
        pair = (
            [6, source[1] + 1, source[0] + 1],
            [6, destination[1] + 1, destination[0] + 1],
        )
        for action in pair:
            play(env, action)
            actions.append(action)
    return actions


def physical_key(env):
    frame = np.asarray(env.frame()).copy()
    frame[0, 0] = 0
    return frame.tobytes()


def probe(env):
    with open("checkpoint.json") as checkpoint_file:
        prefix = json.load(checkpoint_file)["final_path"]
    with open("level9_candidate_102.json") as candidate_file:
        inherited = json.load(candidate_file)
    for action in prefix:
        play(env, action)
    entry = env.clone()
    goals, states = goal_paths(entry.frame())
    print("ENUM", {
        "states": states,
        "goals": len(goals),
        "depths": dict(sorted(Counter(map(len, goals)).items())),
    }, flush=True)
    worlds = {}
    invalid = 0
    for path in goals:
        node = entry.clone()
        entry_actions = play_entry_path(node, path)
        if extract(node.frame())[0]:
            invalid += 1
            continue
        key = physical_key(node)
        previous = worlds.get(key)
        if previous is None or len(entry_actions) < len(previous[0]):
            worlds[key] = (entry_actions, node)
    print("WORLDS", {
        "distinct": len(worlds),
        "invalid": invalid,
        "entry_lengths": dict(sorted(Counter(
            len(actions) for actions, _ in worlds.values()
        ).items())),
    }, flush=True)
    winners = []
    suffix_tail = inherited[37:]
    for world_index, (entry_actions, world) in enumerate(worlds.values(), 1):
        for right_count in range(10):
            node = world.clone()
            trial_suffix = [4] * right_count + suffix_tail
            executed = []
            for action in trial_suffix:
                play(node, action)
                executed.append(action)
                if node.levels_completed > 8:
                    break
            if node.levels_completed > 8:
                candidate = entry_actions + executed
                winners.append(candidate)
                print("WIN", {
                    "world": world_index,
                    "entry": len(entry_actions),
                    "right_count": right_count,
                    "suffix": len(executed),
                    "total": len(candidate),
                }, flush=True)
    if winners:
        best = min(winners, key=len)
        with open("level9_entry_variant_candidate.json", "w") as candidate_file:
            json.dump(best, candidate_file, indent=2)
            candidate_file.write("\n")
    print("RESULT", {
        "worlds": len(worlds),
        "winners": len(winners),
        "best": min(map(len, winners)) if winners else None,
    })


levels, path, error = arena.run_program("lf52", probe)
print("HARNESS", {
    "levels": levels,
    "moves": len(path),
    "error": str(error) if error else None,
})

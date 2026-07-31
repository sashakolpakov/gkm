"""Enumerate distinct shortest post-entry worlds without modifying candidates."""

from collections import Counter, deque
import json
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from legs import _bridge_carrier_state


def play(env, action):
    if isinstance(action, list):
        env.step(*action)
    else:
        env.step(action)


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
    return (
        frozenset(destinations),
        frozenset(bridges),
        frozenset(pegs),
        frozenset(carriers),
    )


def goal_paths(frame, max_depth=18):
    destinations, bridges, pegs, carriers = extract(frame)
    start = (bridges, pegs)
    queue = deque([(start, ())])
    seen = {start}
    goals = []
    while queue:
        (state_bridges, state_pegs), path = queue.popleft()
        if len(state_pegs) == 1 and state_pegs <= carriers:
            goals.append(path)
            continue
        if len(path) >= max_depth:
            continue
        occupied = state_bridges | state_pegs
        for kind, pieces in (("bridge", state_bridges), ("peg", state_pegs)):
            for source in sorted(pieces):
                for dr, dc in ((-6, 0), (6, 0), (0, -6), (0, 6)):
                    midpoint = (source[0] + dr, source[1] + dc)
                    destination = (
                        source[0] + 2 * dr,
                        source[1] + 2 * dc,
                    )
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
                    child = (
                        frozenset(child_bridges),
                        frozenset(child_pegs),
                    )
                    if child in seen:
                        continue
                    seen.add(child)
                    queue.append((
                        child,
                        path + ((kind, source, destination),),
                    ))
    return goals, len(seen)


def actions_for(path):
    actions = []
    for _, source, destination in path:
        actions.extend((
            [6, source[1] + 1, source[0] + 1],
            [6, destination[1] + 1, destination[0] + 1],
        ))
    return actions


def physical_key(env):
    frame = np.asarray(env.frame()).copy()
    frame[0, :] = 0
    return frame.tobytes()


def summary(env):
    slots, pegs, carriers, bridges, borders, selected = (
        _bridge_carrier_state(env.frame())
    )
    return {
        "slots": len(slots),
        "pegs": sorted(pegs),
        "carriers": sorted(carriers),
        "bridges15": sorted(bridges),
        "borders": sorted(borders),
        "selected": selected,
    }


def probe(env):
    with open("checkpoint.json") as checkpoint_file:
        prefix = json.load(checkpoint_file)["final_path"]
    for action in prefix:
        play(env, action)
    entry = env.clone()
    goals, seen = goal_paths(entry.frame())
    shortest = min(map(len, goals))
    goals = [path for path in goals if len(path) == shortest]
    worlds = {}
    for path in goals:
        node = entry.clone()
        actions = actions_for(path)
        for action in actions:
            play(node, action)
        key = physical_key(node)
        worlds.setdefault(key, (actions, node, path))
    print("ENUM", {
        "states": seen,
        "goal_depths": dict(Counter(map(len, goals))),
        "shortest_goals": len(goals),
        "worlds": len(worlds),
    }, flush=True)
    for index, (_, (actions, node, path)) in enumerate(worlds.items()):
        print("WORLD", {
            "index": index,
            "actions": actions,
            "abstract_path": path,
            "summary": summary(node),
        }, flush=True)


levels, path, error = arena.run_program("lf52", probe)
print("HARNESS", {"levels": levels, "moves": len(path), "error": str(error) if error else None})

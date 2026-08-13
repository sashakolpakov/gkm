"""Enumerate post-entry worlds while preserving visually hidden piece identity."""

from collections import Counter, deque
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from legs import _bridge_carrier_state
from perception import frame_delta
from probe_level9_shortest_suffix import dense_summary, lattice, search


MAX_DEPTH = int(os.environ.get("LABELED_MAX_DEPTH", "18"))
MAX_GOALS = int(os.environ.get("LABELED_MAX_GOALS", "500"))
SEARCH_WORLD = os.environ.get("SEARCH_WORLD")


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
    return destinations, sorted(bridges), sorted(pegs), carriers


def labeled_goals(frame):
    destinations, start_bridges, start_pegs, carriers = extract(frame)
    start = (tuple(start_bridges), tuple(start_pegs))
    queue = deque([(start, ())])
    seen = {start}
    goals = []
    while queue and len(goals) < MAX_GOALS:
        (bridge_positions, peg_positions), path = queue.popleft()
        live_pegs = {position for position in peg_positions if position is not None}
        if len(live_pegs) == 1 and live_pegs <= carriers:
            goals.append(((bridge_positions, peg_positions), path))
            continue
        if len(path) >= MAX_DEPTH:
            continue
        occupied = set(bridge_positions) | live_pegs
        labeled_pieces = (
            tuple(("bridge", index, position)
                  for index, position in enumerate(bridge_positions))
            + tuple(("peg", index, position)
                    for index, position in enumerate(peg_positions)
                    if position is not None)
        )
        for kind, index, source in labeled_pieces:
            for dr, dc in ((-6, 0), (6, 0), (0, -6), (0, 6)):
                midpoint = (source[0] + dr, source[1] + dc)
                destination = (source[0] + 2 * dr, source[1] + 2 * dc)
                if (
                    midpoint not in occupied
                    or destination not in destinations
                    or destination in occupied
                ):
                    continue
                child_bridges = list(bridge_positions)
                child_pegs = list(peg_positions)
                if kind == "bridge":
                    child_bridges[index] = destination
                else:
                    child_pegs[index] = destination
                    for jumped_index, jumped in enumerate(child_pegs):
                        if jumped_index != index and jumped == midpoint:
                            child_pegs[jumped_index] = None
                child = (tuple(child_bridges), tuple(child_pegs))
                if child in seen:
                    continue
                seen.add(child)
                queue.append((
                    child,
                    path + ((kind, index, source, destination),),
                ))
    return goals, len(seen), start


def actions_for(path):
    actions = []
    for _, _, source, destination in path:
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
    goals, seen, start = labeled_goals(entry.frame())
    worlds = {}
    for terminal, path in goals:
        node = entry.clone()
        actions = actions_for(path)
        for action in actions:
            play(node, action)
        key = physical_key(node)
        record = worlds.setdefault(key, {
            "paths": 0,
            "best": None,
            "node": node,
            "terminal": terminal,
        })
        record["paths"] += 1
        if record["best"] is None or len(actions) < len(record["best"]):
            record["best"] = actions
            record["terminal"] = terminal
            record["node"] = node
    print("ENUM", {
        "start": start,
        "states": seen,
        "goals": len(goals),
        "goal_depths": dict(Counter(len(path) for _, path in goals)),
        "worlds": len(worlds),
    }, flush=True)
    ordered_worlds = sorted(
        worlds.values(), key=lambda item: len(item["best"])
    )
    reference_frame = np.asarray(ordered_worlds[0]["node"].frame()).copy()
    reference_frame[0, :] = 0
    for index, record in enumerate(ordered_worlds):
        compared_frame = np.asarray(record["node"].frame()).copy()
        compared_frame[0, :] = 0
        holes, movable, pegs, carriers, supports = lattice(compared_frame)
        print("WORLD", {
            "index": index,
            "paths": record["paths"],
            "cost": len(record["best"]),
            "terminal": record["terminal"],
            "actions": record["best"],
            "summary": summary(record["node"]),
            "lattice": {
                "holes": len(holes),
                "movable": sorted(movable),
                "pegs": sorted(pegs),
                "carriers": sorted(carriers),
                "supports": sorted(supports),
            },
            "delta_from_0": {
                key: value
                for key, value in frame_delta(
                    reference_frame, compared_frame
                ).items()
                if key != "samples"
            },
        }, flush=True)
    if SEARCH_WORLD is not None:
        index = int(SEARCH_WORLD)
        record = ordered_worlds[index]
        result = search(record["node"])
        print("WORLD_SEARCH", {
            "index": index,
            "entry_cost": len(record["best"]),
            "entry_actions": record["best"],
            "dense": dense_summary(record["node"]),
            "result": result,
        }, flush=True)


levels, path, error = arena.run_program("lf52", probe)
print("HARNESS", {"levels": levels, "moves": len(path), "error": str(error) if error else None})

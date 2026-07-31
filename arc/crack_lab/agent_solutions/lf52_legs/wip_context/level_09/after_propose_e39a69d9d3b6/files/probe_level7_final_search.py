"""Best-first macro search for a shorter level-7 final-board solution."""

from heapq import heappop, heappush
import json
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from legs import _movable_bridge_board


FINAL_ENTRY_INDEX = 446
CURRENT_FINAL_COST = 30
MAX_STATES = 1800


def play(env, action):
    if isinstance(action, tuple):
        env.step(*action)
    else:
        env.step(action)


def physical_key(env):
    frame = np.asarray(env.frame()).copy()
    frame[0, :] = 0
    return frame.tobytes()


def lattice_actions(env):
    slots, carriers, bridges, pegs = _movable_bridge_board(env.frame())
    destinations = slots | carriers
    occupied = pegs | bridges
    for kind, pieces in (("peg", pegs), ("bridge", bridges)):
        for source in sorted(pieces):
            for dr, dc in ((-6, 0), (6, 0), (0, -6), (0, 6)):
                midpoint = (source[0] + dr, source[1] + dc)
                destination = (source[0] + 2 * dr, source[1] + 2 * dc)
                if (
                    midpoint not in occupied
                    or destination not in destinations
                    or destination in occupied
                    or (kind == "bridge" and midpoint not in pegs)
                ):
                    continue
                yield (
                    (6, source[1] + 1, source[0] + 1),
                    (6, destination[1] + 1, destination[0] + 1),
                )


def heuristic(env):
    _, _, _, pegs = _movable_bridge_board(env.frame())
    if len(pegs) <= 1:
        return 0
    distances = [
        (abs(first[0] - second[0]) + abs(first[1] - second[1])) // 6
        for index, first in enumerate(sorted(pegs))
        for second in sorted(pegs)[index + 1:]
    ]
    return 2 + max(0, min(distances) - 1)


def search(entry):
    serial = 0
    start_key = physical_key(entry)
    queue = [(heuristic(entry), 0, serial, entry.clone(), ())]
    best_cost = {start_key: 0}
    expanded = 0
    while queue and expanded < MAX_STATES:
        _, cost, _, state, path = heappop(queue)
        key = physical_key(state)
        if cost != best_cost.get(key):
            continue
        expanded += 1
        if expanded % 100 == 0:
            print("PROGRESS", {
                "expanded": expanded,
                "queue": len(queue),
                "cost": cost,
                "pieces": _movable_bridge_board(state.frame())[1:],
            }, flush=True)
        edges = [((action,), 1) for action in (1, 2, 3, 4)]
        edges.extend((actions, 2) for actions in lattice_actions(state))
        for actions, edge_cost in edges:
            child_cost = cost + edge_cost
            if child_cost >= CURRENT_FINAL_COST:
                continue
            child = state.clone()
            for action in actions:
                play(child, action)
            child_path = path + actions
            if child.levels_completed > 6:
                return child_path, expanded
            child_key = physical_key(child)
            if child_cost >= best_cost.get(child_key, CURRENT_FINAL_COST):
                continue
            best_cost[child_key] = child_cost
            serial += 1
            heappush(
                queue,
                (
                    child_cost + heuristic(child),
                    child_cost,
                    serial,
                    child,
                    child_path,
                ),
            )
    return None, expanded


def probe(env):
    with open("checkpoint.json") as checkpoint_file:
        campaign = json.load(checkpoint_file)["final_path"]
    for action in campaign[:FINAL_ENTRY_INDEX]:
        if isinstance(action, list):
            env.step(*action)
        else:
            env.step(action)
    entry = env.clone()
    print("ENTRY", {
        "level": entry.levels_completed,
        "board": _movable_bridge_board(entry.frame()),
    }, flush=True)
    result, expanded = search(entry)
    if result is not None:
        candidate = campaign[331:FINAL_ENTRY_INDEX] + [
            list(action) if isinstance(action, tuple) else action
            for action in result
        ]
        with open("level7_final_search_candidate.json", "w") as candidate_file:
            json.dump(candidate, candidate_file, indent=2)
            candidate_file.write("\n")
    print("RESULT", {
        "found": result is not None,
        "final_cost": len(result) if result is not None else None,
        "level_cost": (
            FINAL_ENTRY_INDEX - 331 + len(result)
            if result is not None else None
        ),
        "expanded": expanded,
    })


levels, path, error = arena.run_program("lf52", probe)
print("HARNESS", {
    "levels": levels,
    "moves": len(path),
    "error": str(error) if error else None,
})

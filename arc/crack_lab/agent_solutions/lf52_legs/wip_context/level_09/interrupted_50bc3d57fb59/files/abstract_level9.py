"""Pure symbolic shortest-path probe for the verified level-9 mechanics."""

from heapq import heappop, heappush
import json
import sys

import numpy as np

sys.path.insert(0, "/private/tmp/gkm-submitted-protocol-lf52-20260805/arc/crack_lab")

import gkm_arena as arena

from perception import arr, connected_components, safe_step


def square_positions(frame, color):
    return frozenset(
        blob.top_left
        for blob in connected_components(frame, colors=(color,))
        if blob.size == (4, 4)
    )


def bridge_positions(frame):
    frame = arr(frame)
    return frozenset(
        (row, col)
        for row in range(0, 61, 6)
        for col in range(0, 61, 6)
        if np.count_nonzero(frame[row:row + 4, col:col + 4] == 9) == 12
        and np.count_nonzero(frame[row:row + 4, col:col + 4] == 1) == 4
    )


def successors(state, board, carrier_track):
    pegs, bridges, carrier = state
    occupied = pegs | bridges
    destinations = (board - occupied) | ({carrier} if carrier not in occupied else set())
    pieces = tuple(("peg", position) for position in sorted(pegs))
    pieces += tuple(("bridge", position) for position in sorted(bridges))
    for kind, source in pieces:
        for dr, dc in ((-6, 0), (6, 0), (0, -6), (0, 6)):
            midpoint = source[0] + dr, source[1] + dc
            destination = source[0] + 2 * dr, source[1] + 2 * dc
            if midpoint not in occupied or destination not in destinations:
                continue
            child_pegs = set(pegs)
            child_bridges = set(bridges)
            if kind == "peg":
                child_pegs.remove(source)
                child_pegs.add(destination)
                if midpoint in child_pegs:
                    child_pegs.remove(midpoint)
            else:
                child_bridges.remove(source)
                child_bridges.add(destination)
            clicks = (
                (6, source[1] + 1, source[0] + 1),
                (6, destination[1] + 1, destination[0] + 1),
            )
            yield (frozenset(child_pegs), frozenset(child_bridges), carrier), clicks

    index = carrier_track.index(carrier)
    for offset, action in ((-1, 3), (1, 4)):
        next_index = index + offset
        if not (0 <= next_index < len(carrier_track)):
            continue
        destination = carrier_track[next_index]
        child_pegs = set(pegs)
        child_bridges = set(bridges)
        if carrier in child_pegs:
            if destination in occupied:
                continue
            child_pegs.remove(carrier)
            child_pegs.add(destination)
        elif carrier in child_bridges:
            if destination in occupied:
                continue
            child_bridges.remove(carrier)
            child_bridges.add(destination)
        yield (frozenset(child_pegs), frozenset(child_bridges), destination), (action,)


def probe(env):
    with open("checkpoint.json") as handle:
        checkpoint = json.load(handle)
    for action in checkpoint["final_path"]:
        env.step(action)
    base_level = int(env.levels_completed)
    frame = env.frame()
    pegs = square_positions(frame, 14)
    bridges = bridge_positions(frame)
    empty_slots = frozenset(
        blob.top_left
        for blob in connected_components(frame, colors=(1,))
        if blob.size == (4, 4) and blob.area == 16
    )
    carriers = square_positions(frame, 12)
    carrier = next(iter(carriers))
    board = empty_slots | pegs | bridges
    carrier_track = tuple((carrier[0], col) for col in range(carrier[1], 61, 6))
    start = (pegs, bridges, carrier)

    queue = [(0, 0, start)]
    best = {start: 0}
    parent = {}
    serial = 0
    goals = []
    while queue and len(best) <= 1_000_000:
        cost, _, state = heappop(queue)
        if cost != best.get(state):
            continue
        if len(state[0]) == 1 and state[2] in state[0]:
            goals.append(state)
            continue
        if cost >= 28:
            continue
        for child, macro in successors(state, board, carrier_track):
            child_cost = cost + len(macro)
            if child_cost > 28 or child_cost >= best.get(child, 10 ** 9):
                continue
            best[child] = child_cost
            parent[child] = state, macro
            serial += 1
            heappush(queue, (child_cost, serial, child))

    goal = goals[0] if goals else None
    actions = None
    if goal is not None:
        macros = []
        state = goal
        while state != start:
            state, macro = parent[state]
            macros.append(macro)
        actions = tuple(action for macro in reversed(macros) for action in macro)
        clone = env.clone()
        for action in actions:
            safe_step(clone, action)
        print(
            "ABSTRACT_REPLAY", clone.levels_completed, len(actions),
            square_positions(clone.frame(), 14), bridge_positions(clone.frame()),
        )
    print("ABSTRACT", "states", len(best), "goal_cost", None if goal is None else best[goal])
    print(
        "GOALS", len(goals),
        tuple((best[state], tuple(sorted(state[1])), state[2]) for state in goals),
    )
    print("ACTIONS", actions)
    print("BASE_LEVEL", base_level)


levels, path, error = arena.run_program("lf52", probe)
print("PROBE_RESULT", levels, len(path), error)

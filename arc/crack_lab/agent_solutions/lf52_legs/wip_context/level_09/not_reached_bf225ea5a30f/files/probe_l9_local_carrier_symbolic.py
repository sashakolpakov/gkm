"""Exact local-board search that permits the level-9 carrier to move."""

import heapq
import os


DIRECTIONS = ((-6, 0), (6, 0), (0, -6), (0, 6))
SLOTS = frozenset({
    *((row, col) for row in (18, 24, 30, 36, 48)
      for col in (-2, 4, 10, 16, 22)),
    *((42, col) for col in (-2, 4, 10, 16)),
})
START = (
    22,
    frozenset({(42, -2), (48, 22)}),
    frozenset({(18, 10), (24, 16), (42, 4), (48, 4)}),
)
ANY_PEG = os.environ.get("ANY_PEG", "0") == "1"


def visible(point):
    return 0 <= point[0] <= 60 and 0 <= point[1] + 20 <= 60


def transitions(state):
    carrier_col, pegs, bridges = state
    occupied = pegs | bridges
    for delta, action in ((-6, 3), (6, 4)):
        child_col = carrier_col + delta
        if not 22 <= child_col <= 46:
            continue
        source = (36, carrier_col)
        destination = (36, child_col)
        child_pegs, child_bridges = set(pegs), set(bridges)
        if source in pegs:
            if destination in occupied - {source}:
                continue
            child_pegs.remove(source)
            child_pegs.add(destination)
        elif source in bridges:
            if destination in occupied - {source}:
                continue
            child_bridges.remove(source)
            child_bridges.add(destination)
        yield ((child_col, frozenset(child_pegs), frozenset(child_bridges)),
               1, action)

    destinations = SLOTS | {(36, carrier_col)}
    for kind, pieces in (("peg", pegs), ("bridge", bridges)):
        for source in sorted(pieces):
            for dr, dc in DIRECTIONS:
                midpoint = (source[0] + dr, source[1] + dc)
                destination = (source[0] + 2 * dr, source[1] + 2 * dc)
                if (
                    midpoint not in occupied
                    or destination not in destinations
                    or destination in occupied
                    or not visible(source)
                    or not visible(destination)
                ):
                    continue
                child_pegs, child_bridges = set(pegs), set(bridges)
                if kind == "peg":
                    child_pegs.remove(source)
                    child_pegs.add(destination)
                    child_pegs.discard(midpoint)
                else:
                    child_bridges.remove(source)
                    child_bridges.add(destination)
                yield ((carrier_col, frozenset(child_pegs),
                        frozenset(child_bridges)),
                       2, (kind, source, destination))


queue = [(0, 0, START)]
best = {START: 0}
parent = {}
serial = 0
goal = None
while queue:
    cost, _, state = heapq.heappop(queue)
    if best.get(state) != cost:
        continue
    carrier_col, pegs, _ = state
    if (36, carrier_col) in pegs and (ANY_PEG or len(pegs) == 1):
        goal = state
        break
    for child, edge_cost, action in transitions(state):
        child_cost = cost + edge_cost
        if child_cost >= best.get(child, 10 ** 9):
            continue
        best[child] = child_cost
        parent[child] = (state, action)
        serial += 1
        heapq.heappush(queue, (child_cost, serial, child))

path = []
if goal is not None:
    state = goal
    while state != START:
        state, action = parent[state]
        path.append(action)
    path.reverse()
print("LOCAL_CARRIER", {
    "any_peg": ANY_PEG,
    "states": len(best),
    "cost": None if goal is None else best[goal],
    "goal": goal,
    "path": path,
})

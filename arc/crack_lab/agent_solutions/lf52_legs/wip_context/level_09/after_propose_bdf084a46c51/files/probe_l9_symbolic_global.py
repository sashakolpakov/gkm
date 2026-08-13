"""Exact weighted search over the reproduced level-9 relay geometry."""

import heapq
import os


REMOTE_SLOTS = frozenset({
    *((12, col) for col in (52, 58, 64, 70, 76, 82, 88, 94,
                             100, 106, 112, 118)),
    *((18, col) for col in range(52, 119, 6)),
    *((24, col) for col in range(52, 119, 6)),
    (30, 112), (30, 118), (36, 112), (36, 118),
})
LOCAL_SLOTS = frozenset({
    *((18, col) for col in (4, 10, 16, 22)),
    *((24, col) for col in (4, 10, 16, 22)),
    *((30, col) for col in (4, 10, 16, 22)),
    *((36, col) for col in (4, 10, 16, 22)),
    *((42, col) for col in (4, 10, 16)),
    *((48, col) for col in (4, 10, 16, 22)),
})
FIXED = frozenset({(12, 70), (12, 82), (24, 58),
                   (24, 94), (24, 106)})
DIRECTIONS = ((-6, 0), (6, 0), (0, -6), (0, 6))


include_local = os.environ.get("INCLUDE_LOCAL", "1") == "1"
slots = REMOTE_SLOTS | (LOCAL_SLOTS if include_local else frozenset())
bridges = {(18, 106), (18, 112)}
if include_local:
    bridges |= {(18, 10), (24, 4), (36, 16), (42, 16)}
start = (22, frozenset({(12, 52), (36, 22)}), frozenset(bridges))
cost_limit = int(os.environ.get("COST_LIMIT", "74"))
state_limit = int(os.environ.get("STATE_LIMIT", "2000000"))


def visible(point, carrier_col):
    screen_col = point[1] - (carrier_col - 22)
    return 0 <= point[0] <= 60 and 0 <= screen_col <= 60


def transitions(state):
    carrier_col, pegs, movable = state
    occupied = pegs | movable | FIXED

    for delta, action in ((-6, 3), (6, 4)):
        child_col = carrier_col + delta
        if not 22 <= child_col <= 106:
            continue
        source = (36, carrier_col)
        destination = (36, child_col)
        child_pegs, child_movable = set(pegs), set(movable)
        if source in pegs:
            if destination in occupied - {source}:
                continue
            child_pegs.remove(source)
            child_pegs.add(destination)
        elif source in movable:
            if destination in occupied - {source}:
                continue
            child_movable.remove(source)
            child_movable.add(destination)
        yield (child_col, frozenset(child_pegs),
               frozenset(child_movable)), 1, action

    destinations = slots | {(36, carrier_col)}
    pieces = tuple(("peg", point) for point in sorted(pegs))
    pieces += tuple(("bridge", point) for point in sorted(movable))
    for kind, source in pieces:
        for dr, dc in DIRECTIONS:
            midpoint = (source[0] + dr, source[1] + dc)
            destination = (source[0] + 2 * dr, source[1] + 2 * dc)
            if (
                midpoint not in occupied
                or destination not in destinations
                or destination in occupied
                or not visible(source, carrier_col)
                or not visible(destination, carrier_col)
            ):
                continue
            child_pegs, child_movable = set(pegs), set(movable)
            if kind == "peg":
                child_pegs.remove(source)
                child_pegs.add(destination)
                if midpoint in child_pegs:
                    child_pegs.remove(midpoint)
            else:
                child_movable.remove(source)
                child_movable.add(destination)
            yield (
                (carrier_col, frozenset(child_pegs),
                 frozenset(child_movable)),
                2,
                (kind, source, destination),
            )


queue = [(0, 0, start)]
best = {start: 0}
parent = {}
serial = 0
goal = None
while queue and len(best) <= state_limit:
    cost, _, state = heapq.heappop(queue)
    if best.get(state) != cost:
        continue
    if len(state[1]) == 1:
        goal = state
        break
    for child, edge_cost, action in transitions(state):
        child_cost = cost + edge_cost
        if child_cost > cost_limit or child_cost >= best.get(child, 10 ** 9):
            continue
        best[child] = child_cost
        parent[child] = (state, action)
        serial += 1
        heapq.heappush(queue, (child_cost, serial, child))

if goal is None:
    print("SYMBOLIC_RESULT", {"include_local": include_local,
                              "states": len(best), "goal": None,
                              "remaining": len(queue)})
else:
    path = []
    state = goal
    while state != start:
        state, action = parent[state]
        path.append(action)
    path.reverse()
    print("SYMBOLIC_RESULT", {"include_local": include_local,
                              "states": len(best), "cost": best[goal],
                              "path": path})

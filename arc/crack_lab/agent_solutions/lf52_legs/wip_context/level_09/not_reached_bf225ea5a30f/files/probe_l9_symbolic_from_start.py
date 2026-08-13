"""Exact level-9 search from the pristine local board through the relay."""

import heapq
import os


DIRECTIONS = ((-6, 0), (6, 0), (0, -6), (0, 6))
LOCAL_SLOTS = frozenset({
    *((row, col) for row in (18, 24, 30, 36, 48)
      for col in (-2, 4, 10, 16, 22)),
    *((42, col) for col in (-2, 4, 10, 16)),
})
REMOTE_SLOTS = frozenset({
    *((12, col) for col in (52, 58, 64, 70, 76, 82, 88, 94,
                             100, 106, 112, 118)),
    *((18, col) for col in range(52, 119, 6)),
    *((24, col) for col in range(52, 119, 6)),
    (30, 112), (30, 118), (36, 112), (36, 118),
})
FIXED = frozenset({(12, 70), (12, 82), (24, 58),
                   (24, 94), (24, 106)})
START = (
    "local", 22,
    frozenset({(42, -2), (48, 22)}),
    frozenset({(18, 10), (24, 16), (42, 4), (48, 4)}),
)
COST_LIMIT = int(os.environ.get("COST_LIMIT", "101"))
STATE_LIMIT = int(os.environ.get("STATE_LIMIT", "5000000"))


def local_visible(point):
    return 0 <= point[0] <= 60 and 0 <= point[1] + 20 <= 60


def relay_visible(point, carrier_col):
    screen_col = point[1] - (carrier_col - 22)
    return 0 <= point[0] <= 60 and 0 <= screen_col <= 60


def transitions(state):
    phase, carrier_col, pegs, bridges = state
    occupied = pegs | bridges | (FIXED if phase == "relay" else frozenset())

    if phase == "local" and len(pegs) == 1 and (36, carrier_col) in pegs:
        child = (
            "relay", carrier_col,
            frozenset(set(pegs) | {(12, 52)}),
            frozenset(set(bridges) | {(18, 106), (18, 112)}),
        )
        yield child, 0, ("phase",)
        return

    rail_min, rail_max = (22, 46) if phase == "local" else (22, 106)
    for delta, action in ((-6, 3), (6, 4)):
        child_col = carrier_col + delta
        if not rail_min <= child_col <= rail_max:
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
        yield ((phase, child_col, frozenset(child_pegs),
                frozenset(child_bridges)), 1, action)

    slots = LOCAL_SLOTS
    if phase == "relay":
        slots |= REMOTE_SLOTS
    destinations = slots | {(36, carrier_col)}
    visible = (local_visible if phase == "local"
               else lambda point: relay_visible(point, carrier_col))
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
                yield ((phase, carrier_col, frozenset(child_pegs),
                        frozenset(child_bridges)),
                       2, (kind, source, destination))


queue = [(0, 0, START)]
best = {START: 0}
parent = {}
serial = 0
goal = None
while queue and len(best) <= STATE_LIMIT:
    cost, _, state = heapq.heappop(queue)
    if best.get(state) != cost:
        continue
    if state[0] == "relay" and len(state[2]) == 1:
        goal = state
        break
    for child, edge_cost, action in transitions(state):
        child_cost = cost + edge_cost
        if (child_cost > COST_LIMIT
                or child_cost >= best.get(child, 10 ** 9)):
            continue
        best[child] = child_cost
        parent[child] = (state, action)
        serial += 1
        heapq.heappush(queue, (child_cost, serial, child))

if goal is None:
    print("FROM_START", {"states": len(best), "goal": None,
                         "remaining": len(queue)})
else:
    path = []
    state = goal
    while state != START:
        state, action = parent[state]
        if action != ("phase",):
            path.append(action)
    path.reverse()
    print("FROM_START", {"states": len(best), "cost": best[goal],
                         "goal": goal, "path": path})

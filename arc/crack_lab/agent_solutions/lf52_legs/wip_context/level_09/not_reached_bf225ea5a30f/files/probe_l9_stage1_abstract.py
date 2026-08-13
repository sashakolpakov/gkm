"""Exact symbolic BFS for level 9's reproduced first walled board."""

from collections import deque


SLOTS = frozenset({
    (18, 18), (18, 24), (18, 30), (18, 36), (18, 42),
    (24, 18), (24, 24), (24, 30), (24, 36), (24, 42),
    (30, 18), (30, 24), (30, 30), (30, 36), (30, 42),
    (36, 18), (36, 24), (36, 30), (36, 36), (36, 42),
    (42, 18), (42, 24), (42, 30), (42, 36),
    (48, 18), (48, 24), (48, 30), (48, 36), (48, 42),
})
START = (
    frozenset({(42, 18), (48, 42)}),
    frozenset({(18, 30), (24, 36), (42, 24), (48, 24)}),
)
TARGET = (36, 42)
DIRECTIONS = ((-6, 0), (6, 0), (0, -6), (0, 6))


def successors(state):
    pegs, bridges = state
    occupied = pegs | bridges
    pieces = tuple(("peg", point) for point in sorted(pegs))
    pieces += tuple(("bridge", point) for point in sorted(bridges))
    for kind, source in pieces:
        for dr, dc in DIRECTIONS:
            midpoint = (source[0] + dr, source[1] + dc)
            destination = (source[0] + 2 * dr, source[1] + 2 * dc)
            if (
                midpoint not in occupied
                or destination not in SLOTS
                or destination in occupied
            ):
                continue
            child_pegs, child_bridges = set(pegs), set(bridges)
            if kind == "peg":
                child_pegs.remove(source)
                child_pegs.add(destination)
                if midpoint in child_pegs:
                    child_pegs.remove(midpoint)
            else:
                child_bridges.remove(source)
                child_bridges.add(destination)
            yield (
                (frozenset(child_pegs), frozenset(child_bridges)),
                (kind, source, destination),
            )


def solve(goal):
    queue = deque([(START, ())])
    seen = {START}
    while queue:
        state, path = queue.popleft()
        if goal(state):
            return path, state, len(seen)
        for child, move in successors(state):
            if child not in seen:
                seen.add(child)
                queue.append((child, path + (move,)))
    return None, None, len(seen)


for name, goal in (
    ("bridge_loaded", lambda state: TARGET in state[1]),
    ("any_loaded", lambda state: TARGET in state[0]),
    ("one_loaded", lambda state: state[0] == frozenset({TARGET})),
):
    solution, state, states = solve(goal)
    print("STAGE1_BFS", {"goal": name, "states": states,
                         "macros": None if solution is None else len(solution),
                         "state": state, "solution": solution})

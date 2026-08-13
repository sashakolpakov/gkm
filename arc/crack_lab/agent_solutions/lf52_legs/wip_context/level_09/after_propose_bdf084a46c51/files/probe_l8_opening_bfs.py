"""Exact symbolic BFS for the reproduced aligned level-8 opening."""

from collections import deque


HOLES = frozenset({
    *((18, col) for col in range(12, 55, 6)),
    (24, 24), (24, 30), (24, 36), (24, 48), (24, 54),
    (30, 12), (30, 18), (30, 24), (30, 30), (30, 36),
    (30, 42), (30, 54),
    (36, 24), (36, 30), (36, 36), (36, 48), (36, 54),
    (42, 12), (42, 18), (42, 24), (42, 36), (42, 42),
    (42, 48), (42, 54),
    (48, 12), (48, 18), (48, 24), (48, 36), (48, 42),
    (48, 48),
})
FIXED = frozenset({(24, 18), (24, 42), (30, 48),
                   (36, 18), (36, 42)})
CARRIERS = frozenset({(36, 12), (60, 54)})
START = (frozenset({(24, 12), (48, 54)}),
         frozenset({(42, 30), (48, 30)}))
SLOTS = HOLES | FIXED | CARRIERS | START[0] | START[1]
DIRECTIONS = ((-6, 0), (6, 0), (0, -6), (0, 6))


def successors(state):
    pegs, bridges = state
    occupied = pegs | bridges | FIXED
    pieces = tuple(("peg", point) for point in sorted(pegs))
    pieces += tuple(("bridge", point) for point in sorted(bridges))
    for kind, source in pieces:
        for dr, dc in DIRECTIONS:
            midpoint = (source[0] + dr, source[1] + dc)
            destination = (source[0] + 2 * dr, source[1] + 2 * dc)
            if (midpoint not in occupied or destination not in SLOTS
                    or destination in occupied):
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
            yield ((frozenset(child_pegs), frozenset(child_bridges)),
                   (kind, source, destination))


def solve(target):
    queue = deque([(START, ())])
    seen = {START}
    while queue:
        state, path = queue.popleft()
        if target in state[0]:
            return path, len(seen)
        for child, move in successors(state):
            if child not in seen:
                seen.add(child)
                queue.append((child, path + (move,)))
    return None, len(seen)


for target in ((36, 12), (60, 54)):
    solution, states = solve(target)
    print("L8_OPENING", {"target": target, "states": states,
                         "macros": None if solution is None else len(solution),
                         "solution": solution})

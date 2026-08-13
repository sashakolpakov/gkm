"""Enumerate all shortest peg/bridge carrier assignments in level 6."""

from collections import deque


SLOTS = frozenset({
    (12, 12), (12, 18), (12, 24), (12, 30),
    (18, 12), (18, 24), (18, 30),
    (24, 12), (24, 24), (24, 30),
    (30, 12), (30, 18), (30, 24), (30, 30),
    (36, 12), (36, 18), (36, 24), (36, 30), (36, 36),
    (36, 42),
    (42, 12), (42, 18), (42, 24), (42, 30), (42, 36),
    (42, 42),
    (48, 18), (48, 24), (48, 30), (48, 36), (48, 42),
})
CARRIERS = frozenset({(42, 48), (42, 54)})
START = (frozenset({(24, 18), (48, 12), (54, 24)}), (18, 18))
DESTINATIONS = SLOTS | CARRIERS
DIRECTIONS = ((-6, 0), (6, 0), (0, -6), (0, 6))


def successors(state):
    pegs, bridge = state
    occupied = pegs | {bridge}
    pieces = tuple(("peg", point) for point in sorted(pegs))
    pieces += (("bridge", bridge),)
    for kind, source in pieces:
        for dr, dc in DIRECTIONS:
            midpoint = (source[0] + dr, source[1] + dc)
            destination = (source[0] + 2 * dr, source[1] + 2 * dc)
            if (destination not in DESTINATIONS or destination in occupied
                    or midpoint not in occupied
                    or (kind == "bridge" and midpoint not in pegs)):
                continue
            child_pegs, child_bridge = set(pegs), bridge
            if kind == "peg":
                child_pegs.remove(source)
                child_pegs.add(destination)
                if midpoint in child_pegs:
                    child_pegs.remove(midpoint)
            else:
                child_bridge = destination
            yield ((frozenset(child_pegs), child_bridge),
                   (kind, source, destination))


queue = deque([(START, ())])
seen_depth = {START: 0}
goals = {}
max_depth = 16
while queue:
    state, path = queue.popleft()
    pegs, bridge = state
    if len(pegs) == 1 and next(iter(pegs)) in CARRIERS:
        goals.setdefault(state, path)
    if len(path) >= max_depth:
        continue
    for child, move in successors(state):
        depth = len(path) + 1
        if depth < seen_depth.get(child, 10 ** 9):
            seen_depth[child] = depth
            queue.append((child, path + (move,)))

variants = []
for (pegs, bridge), path in goals.items():
    occupied = pegs | {bridge}
    for dr, dc in DIRECTIONS:
        midpoint = (bridge[0] + dr, bridge[1] + dc)
        destination = (bridge[0] + 2 * dr, bridge[1] + 2 * dc)
        if (midpoint in pegs and destination in CARRIERS
                and destination not in occupied):
            variants.append({"peg": next(iter(pegs)),
                             "bridge": destination,
                             "macros": len(path) + 1,
                             "path": path + (("bridge", bridge,
                                               destination),)})

print("L6_VARIANTS", {"max_depth": max_depth,
                      "goals": len(goals), "variants": len(variants)})
for variant in sorted(variants, key=lambda item: item["macros"]):
    print("VARIANT", variant)

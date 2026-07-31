from collections import deque


SHORT = ((18, 6), (12, 6), (12, 0))
LONG = (
    (6, 36), (6, 42), (6, 48), (12, 48),
    *((18, column) for column in range(18, 55, 6)),
    (24, 18), (24, 30), (30, 30),
    (24, 42), (30, 42), (36, 42),
    (24, 54),
)
FIXED = frozenset(((36, 12), (36, 24), (42, 12), (42, 24)))


def key(slots, pegs, bridge, cargos):
    return frozenset(slots), frozenset(pegs), bridge, tuple(cargos)


def total_pegs(state):
    return len(state[1]) + sum(cargo == "peg" for cargo in state[3])


start = key(
    ((24, 6), (36, 6), (36, 30), (36, 54),
     (42, 6), (42, 18), (42, 30)),
    ((30, 6), (42, 54)),
    (36, 18),
    (None, None),
)
queue = deque([(start, ())])
seen = {start}
while queue and len(seen) < 200000:
    state, path = queue.popleft()
    slots, pegs, bridge, cargos = state
    if total_pegs(state) == 1 and "peg" in cargos:
        print("FOUND", len(seen), path, state)
        break

    placements = (SHORT, LONG)
    sources = [("peg", peg, None) for peg in pegs]
    if bridge is not None:
        sources.append(("bridge", bridge, None))
    for carrier, cargo in enumerate(cargos):
        if cargo is not None:
            sources.extend(
                (cargo, position, carrier) for position in placements[carrier]
            )

    destinations = [(slot, None) for slot in slots]
    for carrier, cargo in enumerate(cargos):
        if cargo is None:
            destinations.extend(
                (position, carrier) for position in placements[carrier]
            )

    for kind, source, source_carrier in sources:
        for destination, destination_carrier in destinations:
            dr = destination[0] - source[0]
            dc = destination[1] - source[1]
            if (dr == 0) == (dc == 0) or abs(dr + dc) != 12:
                continue
            midpoint = (source[0] + dr // 2, source[1] + dc // 2)
            supports = set(pegs) | {bridge} | set(FIXED)
            special = (
                kind == "peg"
                and source_carrier == 1
                and source == (24, 54)
                and destination == (36, 54)
            ) or (
                kind == "peg"
                and source_carrier is None
                and source == (36, 54)
                and destination_carrier == 1
                and destination == (24, 54)
            )
            if midpoint not in supports and not special:
                continue

            child_slots = set(slots)
            child_pegs = set(pegs)
            child_bridge = bridge
            child_cargos = list(cargos)
            if source_carrier is None:
                child_slots.add(source)
                if kind == "peg":
                    child_pegs.remove(source)
                else:
                    child_bridge = None
            else:
                child_cargos[source_carrier] = None

            if kind == "peg" and midpoint in child_pegs:
                child_pegs.remove(midpoint)
                child_slots.add(midpoint)

            if destination_carrier is None:
                child_slots.remove(destination)
                if kind == "peg":
                    child_pegs.add(destination)
                else:
                    child_bridge = destination
            else:
                child_cargos[destination_carrier] = kind

            child = key(
                child_slots, child_pegs, child_bridge, child_cargos,
            )
            if child in seen:
                continue
            seen.add(child)
            step = (
                kind, source, destination,
                source_carrier, destination_carrier,
            )
            queue.append((child, path + (step,)))
else:
    print("NOT_FOUND", len(seen))

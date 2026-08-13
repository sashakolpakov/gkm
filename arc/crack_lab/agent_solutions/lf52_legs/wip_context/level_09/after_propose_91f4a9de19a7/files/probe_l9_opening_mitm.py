"""Meet in the middle over every 14-jump level-9 opening layout."""

import itertools
import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from perception import connected_components


def parse(frame):
    blobs = connected_components(frame, colors=(1, 9, 12, 14))
    cells = sorted({
        blob.top_left for blob in blobs
        if blob.size == (4, 4) and blob.color in (1, 9, 12, 14)
    })
    pegs = {
        blob.top_left for blob in blobs
        if blob.color == 14 and blob.size == (4, 4)
    }
    bridges = {
        blob.top_left for blob in blobs
        if blob.color == 9 and blob.size == (4, 4) and blob.area == 12
    }
    carrier = next(
        blob.top_left for blob in blobs
        if blob.color == 12 and blob.size == (4, 4)
    )
    return cells, pegs, bridges, carrier


def probe(env):
    with open("checkpoint.json") as stream:
        campaign = json.load(stream)["final_path"]
    for action in campaign:
        env.step(action)
    cells, pegs, bridges, carrier = parse(env.frame())
    indexes = {cell: index for index, cell in enumerate(cells)}
    triples = []
    for source in cells:
        for dr, dc in ((-6, 0), (6, 0), (0, -6), (0, 6)):
            midpoint = (source[0] + dr, source[1] + dc)
            destination = (source[0] + 2 * dr, source[1] + 2 * dc)
            if midpoint in indexes and destination in indexes:
                triples.append((indexes[source], indexes[midpoint],
                                indexes[destination]))

    def bits(points):
        value = 0
        for point in points:
            value |= 1 << indexes[point]
        return value

    def point(index):
        return cells[index]

    start = (bits(pegs), bits(bridges))

    def forward_children(state):
        peg_bits, bridge_bits = state
        occupied = peg_bits | bridge_bits
        for source, midpoint, destination in triples:
            source_mask = 1 << source
            midpoint_mask = 1 << midpoint
            destination_mask = 1 << destination
            if not occupied & midpoint_mask or occupied & destination_mask:
                continue
            if peg_bits & source_mask:
                child_pegs = (peg_bits ^ source_mask) | destination_mask
                if peg_bits & midpoint_mask:
                    child_pegs ^= midpoint_mask
                yield ((child_pegs, bridge_bits),
                       (point(source), point(destination)))
            if bridge_bits & source_mask:
                child_bridges = (bridge_bits ^ source_mask) | destination_mask
                yield ((peg_bits, child_bridges),
                       (point(source), point(destination)))

    def reverse_children(state):
        peg_bits, bridge_bits = state
        occupied = peg_bits | bridge_bits
        for source, midpoint, destination in triples:
            source_mask = 1 << source
            midpoint_mask = 1 << midpoint
            destination_mask = 1 << destination
            if occupied & source_mask or not occupied & midpoint_mask:
                continue
            if peg_bits & destination_mask and bridge_bits & midpoint_mask:
                predecessor_pegs = (peg_bits ^ destination_mask) | source_mask
                yield ((predecessor_pegs, bridge_bits),
                       (point(source), point(destination)))
            if bridge_bits & destination_mask:
                predecessor_bridges = (
                    (bridge_bits ^ destination_mask) | source_mask
                )
                yield ((peg_bits, predecessor_bridges),
                       (point(source), point(destination)))
        if peg_bits.bit_count() == 1:
            for source, midpoint, destination in triples:
                source_mask = 1 << source
                midpoint_mask = 1 << midpoint
                destination_mask = 1 << destination
                if (
                    not peg_bits & destination_mask
                    or occupied & (source_mask | midpoint_mask)
                ):
                    continue
                predecessor_pegs = (
                    (peg_bits ^ destination_mask)
                    | source_mask | midpoint_mask
                )
                yield ((predecessor_pegs, bridge_bits),
                       (point(source), point(destination)))

    forward = {start: ()}
    frontier = {start}
    for depth in range(1, 9):
        children = set()
        for state in frontier:
            prefix = forward[state]
            for child, action in forward_children(state):
                if child in forward:
                    continue
                forward[child] = prefix + (action,)
                children.add(child)
        frontier = children
        print("forward", depth, len(frontier), len(forward), flush=True)

    carrier_index = indexes[carrier]
    available = [index for index in range(len(cells))
                 if index != carrier_index]
    backward = {}
    frontier = set()
    carrier_bit = 1 << carrier_index
    for combination in itertools.combinations(available, len(bridges)):
        bridge_bits = 0
        for index in combination:
            bridge_bits |= 1 << index
        goal = (carrier_bit, bridge_bits)
        backward[goal] = ((), goal)
        frontier.add(goal)
    print("goals", len(frontier), flush=True)

    matches = {}
    for depth in range(0, 7):
        for state in frontier:
            if state not in forward:
                continue
            suffix, goal = backward[state]
            path = forward[state] + suffix
            if len(path) <= 14:
                matches.setdefault(goal, path)
        print("backward", depth, len(frontier), len(backward),
              len(matches), flush=True)
        if depth == 6:
            break
        children = set()
        for state in frontier:
            suffix, goal = backward[state]
            for predecessor, action in reverse_children(state):
                if predecessor in backward:
                    continue
                backward[predecessor] = ((action,) + suffix, goal)
                children.add(predecessor)
        frontier = children
    print("matches", len(matches), tuple(
        (tuple(sorted(point(index) for index in range(len(cells))
                      if goal[1] & (1 << index))), path)
        for goal, path in list(matches.items())[:40]
    ), flush=True)


arena.run_program("lf52", probe)

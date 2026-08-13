"""Enumerate and replay level-9 local captures with any surviving peg."""

import json
import os
import sys
from collections import deque

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from perception import arr, connected_components, safe_step


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


def compact(frame):
    return tuple(
        (blob.color, blob.bbox, blob.area)
        for blob in connected_components(frame, colors=(9, 11, 12, 14, 15))
        if blob.color != 9 or blob.area >= 12
    )


def probe(env):
    with open("checkpoint.json") as stream:
        campaign = json.load(stream)["final_path"]
    for action in campaign:
        safe_step(env, tuple(action) if isinstance(action, list) else action)

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

    def points(value):
        return tuple(cells[index] for index in range(len(cells))
                     if value & (1 << index))

    start = (bits(pegs), bits(bridges))
    queue = deque([(start, ())])
    seen = {start}
    goals = []
    max_depth = int(os.environ.get("OPT_DEPTH", "14"))
    while queue:
        state, path = queue.popleft()
        peg_bits, bridge_bits = state
        if peg_bits.bit_count() == 1:
            goals.append((len(path), points(peg_bits), points(bridge_bits),
                          path))
            continue
        if len(path) >= max_depth:
            continue
        occupied = peg_bits | bridge_bits
        for source, midpoint, destination in triples:
            source_mask = 1 << source
            midpoint_mask = 1 << midpoint
            destination_mask = 1 << destination
            if not occupied & midpoint_mask or occupied & destination_mask:
                continue
            candidates = []
            if peg_bits & source_mask:
                child_pegs = (peg_bits ^ source_mask) | destination_mask
                if peg_bits & midpoint_mask:
                    child_pegs ^= midpoint_mask
                candidates.append((child_pegs, bridge_bits))
            if bridge_bits & source_mask:
                candidates.append((peg_bits,
                                   (bridge_bits ^ source_mask)
                                   | destination_mask))
            for child in candidates:
                if child in seen:
                    continue
                seen.add(child)
                move = (cells[source], cells[destination])
                queue.append((child, path + (move,)))

    goals.sort(key=lambda item: (item[0], item[1], item[2]))
    counts = {}
    for depth, *_ in goals:
        counts[depth] = counts.get(depth, 0) + 1
    print("local_goals", len(seen), tuple(sorted(counts.items())),
          "carrier", carrier, flush=True)
    limit = int(os.environ.get("OPT_REPLAY", "40"))
    for index, (depth, goal_pegs, goal_bridges, path) in enumerate(
            goals[:limit]):
        child = env.clone()
        for source, destination in path:
            safe_step(child, (6, source[1] + 1, source[0] + 1))
            safe_step(child, (6, destination[1] + 1,
                              destination[0] + 1))
        base = arr(child.frame())[1:, :].tobytes()
        keys = []
        for action in (1, 2, 3, 4):
            shifted = child.clone()
            safe_step(shifted, action)
            if arr(shifted.frame())[1:, :].tobytes() != base:
                keys.append(action)
        print("local_replay", index, depth, goal_pegs, goal_bridges,
              int(child.levels_completed), tuple(keys), compact(child.frame()),
              path, flush=True)


arena.run_program("lf52", probe)

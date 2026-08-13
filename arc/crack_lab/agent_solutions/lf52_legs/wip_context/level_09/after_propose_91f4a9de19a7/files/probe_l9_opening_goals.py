"""Enumerate distinct one-carried-peg openings for level 9 symbolically."""

import json
import os
import sys
from collections import deque

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from perception import connected_components, safe_step


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


def parse_world(frame, offset):
    blobs = connected_components(frame, colors=(1, 9, 14, 15))
    holes = {
        (blob.top_left[0], blob.top_left[1] + offset)
        for blob in blobs
        if blob.color == 1 and blob.size == (4, 4) and blob.area == 16
    }
    bridges = {
        (blob.top_left[0], blob.top_left[1] + offset)
        for blob in blobs
        if blob.color == 9 and blob.size == (4, 4) and blob.area == 12
    }
    pegs = {
        (blob.top_left[0], blob.top_left[1] + offset)
        for blob in blobs
        if blob.color == 14 and blob.size == (4, 4)
    }
    fixed = {
        (blob.bbox[0] + 1, blob.bbox[1] + offset)
        for blob in blobs
        if blob.color == 15 and blob.size == (4, 4) and blob.area == 12
    }
    return holes, bridges, pegs, fixed


def probe(env):
    with open("checkpoint.json") as stream:
        campaign = json.load(stream)["final_path"]
    for action in campaign:
        safe_step(env, tuple(action) if isinstance(action, list) else action)

    entry = env.clone()

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
    carrier_bit = 1 << indexes[carrier]
    queue = deque([(start, 0)])
    seen = {start}
    parent = {start: (None, None)}
    goals = []
    max_depth = int(os.environ.get("OPT_DEPTH", "22"))
    depth_counts = {0: 1}

    while queue:
        state, depth = queue.popleft()
        peg_bits, bridge_bits = state
        if peg_bits == carrier_bit:
            path = []
            cursor = state
            while parent[cursor][0] is not None:
                cursor, action = parent[cursor]
                path.append(action)
            goals.append((depth, points(bridge_bits), tuple(reversed(path))))
        if depth >= max_depth:
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
                action = (cells[source], cells[destination])
                parent[child] = (state, action)
                queue.append((child, depth + 1))
                depth_counts[depth + 1] = depth_counts.get(depth + 1, 0) + 1

    by_depth = {}
    for depth, layout, path in goals:
        by_depth.setdefault(depth, []).append((layout, path))
    print("counts", tuple(sorted(depth_counts.items())), "states", len(seen),
          flush=True)
    print("goal_counts", tuple((depth, len(items))
                                for depth, items in sorted(by_depth.items())),
          flush=True)
    layout_depth = {}
    for depth, items in sorted(by_depth.items()):
        for layout, _ in items:
            layout_depth.setdefault(layout, depth)
    compact_only = os.environ.get("OPT_COMPACT") == "1"
    if compact_only:
        print("layout_count", len(layout_depth), flush=True)
    else:
        print("layout_depths", tuple(sorted(
            (depth, layout) for layout, depth in layout_depth.items()
        )), flush=True)
    for depth, items in sorted(by_depth.items()):
        if compact_only:
            break
        for layout, path in items[:5]:
            print("goal", depth, layout, path, flush=True)
    if os.environ.get("OPT_SYMBOLIC_SCORE") == "1":
        from probe_l9_world_model import search

        standard_layout, standard_path = by_depth[14][0]
        opened = entry.clone()
        for source, destination in standard_path:
            safe_step(opened, (6, source[1] + 1, source[0] + 1))
            safe_step(opened, (6, destination[1] + 1,
                              destination[0] + 1))
        world_cells = set()
        world_bridges = set()
        world_fixed = set()
        world_pegs = None
        scan = opened.clone()
        for scan_index in range(15):
            offset = scan_index * 6
            holes, visible_bridges, visible_pegs, fixed = parse_world(
                scan.frame(), offset
            )
            world_cells |= holes | visible_bridges | visible_pegs | fixed
            world_bridges |= visible_bridges
            world_fixed |= fixed
            if scan_index == 0:
                world_pegs = visible_pegs
            if scan_index < 14:
                safe_step(scan, 4)

        def shifted(layout):
            return frozenset((row, col - 20) for row, col in layout)

        remote_bridges = frozenset(world_bridges) - shifted(standard_layout)
        representatives = sorted(
            (depth, layout)
            for layout, depth in layout_depth.items()
        )
        batch_start = int(os.environ.get("OPT_BATCH_START", "0"))
        batch_size = int(os.environ.get("OPT_BATCH_SIZE", "10"))
        target_total = int(os.environ.get("OPT_TARGET_TOTAL", "102"))
        max_cost = int(os.environ.get("OPT_SCORE_COST", "74"))
        print("symbolic_batch", batch_start, batch_size,
              len(representatives), tuple(sorted(remote_bridges)), flush=True)
        for item_index, (depth, layout) in enumerate(
                representatives[batch_start:batch_start + batch_size],
                batch_start):
            bridges = remote_bridges | shifted(layout)
            score_limit = min(max_cost, target_total - 1 - 2 * depth)
            suffix, states, _ = search(
                frozenset(world_cells), frozenset(world_fixed),
                frozenset(world_pegs), bridges, max_cost=score_limit,
            )
            suffix_cost = None if suffix is None else sum(
                1 if step[0] in (3, 4) else 2 for step in suffix
            )
            total = None if suffix_cost is None else 2 * depth + suffix_cost
            print("symbolic_score", item_index, depth, score_limit,
                  suffix_cost, total, states, layout,
                  suffix if total is not None else None, flush=True)
    if os.environ.get("OPT_DEEP_SCORE") == "1":
        from probe_l9_world_model import search

        min_depth = int(os.environ.get("OPT_MIN_LAYOUT_DEPTH", "23"))
        batch_start = int(os.environ.get("OPT_BATCH_START", "0"))
        batch_size = int(os.environ.get("OPT_BATCH_SIZE", "2"))
        max_cost = int(os.environ.get("OPT_SCORE_COST", "74"))
        target_total = int(os.environ.get("OPT_TARGET_TOTAL", "70"))
        representatives = {}
        for depth, items in sorted(by_depth.items()):
            for layout, path in items:
                representatives.setdefault(layout, (depth, path))
        selected = sorted(
            (depth, layout, path)
            for layout, (depth, path) in representatives.items()
            if depth >= min_depth
        )
        print("deep_batch", batch_start, batch_size, "of", len(selected),
              flush=True)
        for item_index, (depth, layout, path) in enumerate(
                selected[batch_start:batch_start + batch_size], batch_start):
            opened = entry.clone()
            for source, destination in path:
                safe_step(opened, (6, source[1] + 1, source[0] + 1))
                safe_step(opened, (6, destination[1] + 1,
                                  destination[0] + 1))
            scan = opened.clone()
            world_cells = set()
            world_bridges = set()
            world_fixed = set()
            initial_pegs = None
            for scan_index in range(15):
                offset = scan_index * 6
                holes, visible_bridges, visible_pegs, fixed = parse_world(
                    scan.frame(), offset
                )
                world_cells |= holes | visible_bridges | visible_pegs | fixed
                world_bridges |= visible_bridges
                world_fixed |= fixed
                if scan_index == 0:
                    initial_pegs = visible_pegs
                if scan_index < 14:
                    safe_step(scan, 4)
            score_limit = min(max_cost, target_total - 2 * depth)
            capture_floor = 2 * (len(initial_pegs) - 1)
            if score_limit < capture_floor:
                suffix, states = None, 0
            else:
                suffix, states, _ = search(
                    frozenset(world_cells), frozenset(world_fixed),
                    frozenset(initial_pegs), frozenset(world_bridges),
                    max_cost=score_limit,
                )
            suffix_cost = None if suffix is None else sum(
                1 if step[0] in (3, 4) else 2 for step in suffix
            )
            total = None if suffix_cost is None else 2 * depth + suffix_cost
            print("deep_score", item_index, depth, score_limit,
                  suffix_cost, total, states,
                  layout, tuple(sorted(world_bridges)),
                  suffix if total is not None and total < 102 else None,
                  path if total is not None and total < 102 else None,
                  flush=True)
    if os.environ.get("OPT_WORLD") == "1":
        world_depth = int(os.environ.get("OPT_WORLD_DEPTH", "18"))
        world_variants = {}
        for depth, items in sorted(by_depth.items()):
            if depth > world_depth:
                continue
            for layout, path in items:
                opened = entry.clone()
                for source, destination in path:
                    safe_step(opened, (6, source[1] + 1, source[0] + 1))
                    safe_step(opened, (6, destination[1] + 1,
                                      destination[0] + 1))
                scan = opened.clone()
                world_cells = set()
                world_bridges = set()
                world_fixed = set()
                initial_pegs = None
                for index in range(15):
                    offset = index * 6
                    holes, visible_bridges, visible_pegs, fixed = parse_world(
                        scan.frame(), offset
                    )
                    world_cells |= holes | visible_bridges | visible_pegs | fixed
                    world_bridges |= visible_bridges
                    world_fixed |= fixed
                    if index == 0:
                        initial_pegs = visible_pegs
                    if index < 14:
                        safe_step(scan, 4)
                print("world_goal", depth, layout,
                      tuple(sorted(initial_pegs)),
                      tuple(sorted(world_bridges)),
                      tuple(sorted(world_fixed)), len(world_cells), flush=True)
                variant = frozenset(world_bridges)
                world_variants.setdefault(
                    variant,
                    (depth, layout, path, frozenset(world_cells),
                     frozenset(initial_pegs), frozenset(world_fixed)),
                )
        print("world_variants", len(world_variants), flush=True)
        if os.environ.get("OPT_PRINT_VARIANTS") == "1":
            print("variant_list", tuple(
                (index, details[0], tuple(sorted(bridges)))
                for index, (bridges, details) in enumerate(sorted(
                    world_variants.items(), key=lambda item: item[1][0]
                ))
            ), flush=True)
        if os.environ.get("OPT_SCORE") == "1":
            from probe_l9_world_model import search

            max_cost = int(os.environ.get("OPT_SCORE_COST", "74"))
            target_total = int(os.environ.get("OPT_TARGET_TOTAL", "102"))
            ordered_variants = sorted(
                world_variants.items(), key=lambda item: item[1][0]
            )
            variant_start = int(os.environ.get("OPT_VARIANT_START", "0"))
            variant_size = int(os.environ.get(
                "OPT_VARIANT_SIZE", str(len(ordered_variants))
            ))
            for variant_index, (bridges, details) in enumerate(
                    ordered_variants[
                        variant_start:variant_start + variant_size
                    ], variant_start):
                depth, layout, path, cells, pegs, fixed = details
                score_limit = min(max_cost, target_total - 1 - 2 * depth)
                suffix, states, goal = search(
                    cells, fixed, pegs, bridges, max_cost=score_limit
                )
                suffix_cost = None if suffix is None else sum(
                    1 if step[0] in (3, 4) else 2 for step in suffix
                )
                total = None if suffix_cost is None else 2 * depth + suffix_cost
                print("score", variant_index, depth, score_limit,
                      suffix_cost, total, states,
                      tuple(sorted(bridges)), layout,
                      suffix if total is not None and total < 102 else None,
                      path if total is not None and total < 102 else None,
                      flush=True)


arena.run_program("lf52", probe)

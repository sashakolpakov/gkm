"""Scan all ring/global phases for a new avatar-reachability signature."""
from collections import deque
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

import perception
import solve
from probe_l6_exact_crossings import (
    CENTER,
    TO_CENTER,
    placement_label,
    placements_with_paths,
)
from probe_l6_right import (
    MAIN,
    SELECTOR,
    TOP,
    avatar_position,
    enter_right,
    movement_reach,
)


def static_reach(env):
    frame = perception.arr(env.frame())
    start = avatar_position(env)
    if start is None:
        return ()
    queue = deque([start])
    seen = {start}
    for_position = (
        (-2, 0), (2, 0), (0, -2), (0, 2),
    )
    while queue and len(seen) < 120:
        row, col = queue.popleft()
        for dr, dc in for_position:
            target = row + dr, col + dc
            nr, nc = target
            if target in seen or not (0 <= nr < 62 and 0 <= nc < 40):
                continue
            block = frame[nr:nr + 2, nc:nc + 2]
            support = sum(
                int(value) not in {0, 4, 5, 15}
                for value in block.flat
            )
            if support >= 2:
                seen.add(target)
                queue.append(target)
    return tuple(sorted(seen))


def metric(positions):
    if not positions:
        return (0, None, None, None, None)
    return (
        len(positions),
        min(row for row, _ in positions),
        max(row for row, _ in positions),
        min(col for _, col in positions),
        max(col for _, col in positions),
    )


def exits(env):
    return tuple(
        blob.bbox
        for blob in perception.connected_components(
            env.frame(), colors=(11,), min_area=4
        )
        if blob.area == 4 and blob.size == (2, 2) and blob.bbox[1] < 40
    )


def observe(env):
    solve.solve(env)
    base_level = env.levels_completed
    placements = placements_with_paths(enter_right(env, 3))
    signatures = {}
    checked = 0
    for index, (placement, _) in enumerate(placements):
        centered = placement.clone()
        position = avatar_position(centered)
        if position != CENTER:
            centered.step(TO_CENTER[position])
        for orientation in range(2):
            oriented = centered.clone()
            if orientation:
                oriented.step(4)
                oriented.step(*MAIN)
                oriented.step(3)
            for bridge_phase in range(6):
                staged = oriented.clone()
                for _ in range(bridge_phase):
                    staged.step(*TOP)
                staged.step(*MAIN)
                for selector_offset in range(4):
                    destination = staged.clone()
                    for _ in range(selector_offset):
                        destination.step(*SELECTOR)
                    destination.step(*MAIN)
                    checked += 1
                    if destination.levels_completed > base_level:
                        print(
                            "GLOBAL_REACH_WIN_CONTROL", index, orientation,
                            bridge_phase, selector_offset, flush=True,
                        )
                        return
                    static_metric = metric(static_reach(destination))
                    region = avatar_position(destination)
                    signature = region, static_metric
                    if signature in signatures and not exits(destination):
                        continue
                    signatures[signature] = (
                        index, orientation, bridge_phase, selector_offset
                    )
                    reached, win = movement_reach(destination)
                    exact_positions = tuple(
                        sorted(
                            position for position in reached
                            if position is not None
                        )
                    )
                    exact_metric = metric(exact_positions)
                    print(
                        "GLOBAL_REACH_SIGNATURE",
                        "config", (
                            index, placement_label(placement),
                            orientation, bridge_phase, selector_offset,
                        ),
                        "avatar", region,
                        "static", static_metric,
                        "exact", exact_metric,
                        "win", win, "exits", exits(destination),
                        flush=True,
                    )
                    if win is not None:
                        print(
                            "GLOBAL_REACH_WIN_WALK",
                            index, orientation, bridge_phase,
                            selector_offset, win, flush=True,
                        )
                        return
        print(
            "GLOBAL_REACH_DONE", index, checked,
            "signatures", len(signatures), flush=True,
        )
    print(
        "GLOBAL_REACH_NO_WIN", checked,
        "signatures", len(signatures), flush=True,
    )


arena.run_program("dc22", observe)

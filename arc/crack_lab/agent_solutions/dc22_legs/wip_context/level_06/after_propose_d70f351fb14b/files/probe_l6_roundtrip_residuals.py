"""Detect persistent state after a physical rotator round trip."""
import sys

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
from probe_l6_ring_route import HUB_TO_BRIDGE
from probe_l6_right import (
    MAIN,
    TOP,
    TO_REMOTE_PAD,
    avatar_position,
    enter_right,
)


def physical_roundtrip(placement):
    node = placement.clone()
    position = avatar_position(node)
    if position != CENTER:
        node.step(TO_CENTER[position])
    node.step(*MAIN)
    for action in HUB_TO_BRIDGE:
        node.step(action)
    node.step(*TOP)
    node.step(1)
    node.step(3)
    node.step(*TOP)
    for _ in range(12):
        node.step(1)
    node.step(*MAIN)
    # Sweep the horizontal rotator before returning to its pivot.
    for _ in range(4):
        node.step(3)
    for _ in range(4):
        node.step(4)
    node.step(*MAIN)
    for _ in range(12):
        node.step(2)
    node.step(*TOP)
    node.step(*TOP)
    node.step(4)
    node.step(*TOP)
    node.step(2)
    node.step(*TOP)
    for action in TO_REMOTE_PAD:
        node.step(action)
    node.step(*MAIN)
    return node


def observe(env):
    solve.solve(env)
    placements = placements_with_paths(enter_right(env, 3))
    for index, (placement, _) in enumerate(placements):
        start = placement.clone()
        position = avatar_position(start)
        if position != CENTER:
            start.step(TO_CENTER[position])
        end = physical_roundtrip(placement)
        before = perception.arr(start.frame())[:63]
        after = perception.arr(end.frame())[:63]
        delta = perception.frame_delta(before, after)
        print(
            "PHYSICAL_ROUNDTRIP", index, placement_label(placement),
            avatar_position(end), delta["count"], delta["bbox"],
        )


arena.run_program("dc22", observe)

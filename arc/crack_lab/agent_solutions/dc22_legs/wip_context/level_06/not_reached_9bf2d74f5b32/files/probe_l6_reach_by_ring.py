"""Measure physical avatar components under every movable-ring placement."""
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

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
    avatar_position,
    enter_right,
    movement_reach,
)


def vertical_entry(placement):
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
    return node


def summarize(reached):
    positions = sorted(position for position in reached if position is not None)
    return (
        len(positions),
        min(positions),
        max(positions),
        tuple(position for position in positions if position[1] >= 18),
        None in reached,
    )


def observe(env):
    solve.solve(env)
    placements = placements_with_paths(enter_right(env, 3))
    for index, (placement, _) in enumerate(placements):
        vertical = vertical_entry(placement)
        vertical_reach, vertical_win = movement_reach(vertical)
        horizontal = vertical.clone()
        for _ in range(12):
            horizontal.step(1)
        horizontal.step(*MAIN)
        horizontal_reach, horizontal_win = movement_reach(horizontal)
        print(
            "REACH_BY_RING", index, placement_label(placement),
            "VERT", summarize(vertical_reach), vertical_win,
            "HORIZ", summarize(horizontal_reach), horizontal_win,
            "PIVOT", avatar_position(horizontal),
        )


if __name__ == "__main__":
    arena.run_program("dc22", observe)

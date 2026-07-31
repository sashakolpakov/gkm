"""Follow context-enabled directions for the central movable ring."""
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

import perception
import solve
from probe_l6_right import (
    avatar_position,
    enter_right,
    movement_reach,
)


UP_CONTROL = (6, 50, 34)
MAIN = (6, 50, 26)
SELECTOR = (6, 50, 46)


def dense_dpad_controls(node):
    before = perception.arr(node.frame()).copy()
    responsive = []
    for y in range(30, 44):
        for x in range(44, 58):
            clone = node.clone()
            clone.step(6, x, y)
            delta = perception.frame_delta(before, clone.frame())
            if any(sample[0] < 63 for sample in delta["samples"]):
                responsive.append((x, y))
    return responsive


def observe(env):
    solve.solve(env)
    node = enter_right(env, 3)
    reached, _ = movement_reach(node)
    for action in reached[(56, 34)]:
        node.step(action)
    print("DPAD_BEFORE", dense_dpad_controls(node))
    alternate = node.clone()
    alternate.step(*MAIN)
    print("DPAD_BEFORE_ALTERNATE_MAIN", dense_dpad_controls(alternate))
    node.step(*UP_CONTROL)
    rings = [
        blob.bbox
        for blob in perception.connected_components(
            node.frame(), colors=(8,), min_area=20
        )
        if blob.bbox[1] < 32
    ]
    print(
        "DPAD_AFTER_UP", rings, avatar_position(node),
        dense_dpad_controls(node),
    )
    node.step(*MAIN)
    print("DPAD_AFTER_UP_MAIN", dense_dpad_controls(node))
    node.step(2)
    node.step(*MAIN)
    hub = avatar_position(node)
    node.step(*SELECTOR)
    node.step(*SELECTOR)
    node.step(*MAIN)
    print(
        "OPPOSITE_PARITY_DEST", hub, avatar_position(node),
        node.levels_completed,
    )


if __name__ == "__main__":
    arena.run_program("dc22", observe)

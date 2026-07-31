"""Test the cargo ring as a bridge from the horizontal central rotator."""
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

import solve
import perception
from probe_l6_right import (
    MAIN,
    TOP,
    avatar_position,
    enter_right,
    movement_reach,
)


LEFT = (6, 46, 36)
HUB_TO_BRIDGE = [2, 2, 2, 2, 3, 3, 3, 2, 3]


def observe(env):
    solve.solve(env)
    node = enter_right(env, 3)
    node.step(3)
    node.step(*LEFT)
    node.step(4)
    node.step(*MAIN)
    print("BRIDGE_RING_HUB", avatar_position(node))
    for action in HUB_TO_BRIDGE:
        node.step(action)
    node.step(*TOP)
    node.step(1)
    node.step(3)
    node.step(*TOP)
    for _ in range(12):
        node.step(1)
    print("BRIDGE_RING_CENTER", avatar_position(node))
    node.step(*MAIN)
    reached, win = movement_reach(node)
    print(
        "BRIDGE_RING_REACH",
        sorted(position for position in reached if position is not None),
        "VANISHED", None in reached, "WIN", win,
    )
    if None in reached:
        vanished = node.clone()
        for action in reached[None]:
            vanished.step(action)
        print(
            "BRIDGE_RING_VANISH_PATH", reached[None],
            vanished.levels_completed, vanished.terminal(),
            [
                (blob.bbox, blob.area)
                for blob in perception.connected_components(
                    vanished.frame(), colors=(14,), min_area=1
                )
            ],
        )
    walker = node.clone()
    for step in range(1, 11):
        walker.step(4)
        world_blobs = [
            (blob.bbox, blob.area)
            for blob in perception.connected_components(
                walker.frame(), colors=(14,), min_area=2
            )
            if blob.bbox[1] < 40
        ]
        print(
            "BRIDGE_RING_WALK", step, world_blobs,
            walker.levels_completed,
        )
    aboard = node.clone()
    for _ in range(5):
        aboard.step(4)
    controls = (
        ("UP", (6, 50, 34)),
        ("DOWN", (6, 50, 40)),
        ("LEFT", (6, 46, 36)),
        ("RIGHT", (6, 54, 36)),
    )
    for name, control in controls:
        branch = aboard.clone()
        before_ring = [
            blob.bbox
            for blob in perception.connected_components(
                branch.frame(), colors=(8,), min_area=20
            )
            if blob.bbox[1] < 32
        ]
        branch.step(*control)
        after_ring = [
            blob.bbox
            for blob in perception.connected_components(
                branch.frame(), colors=(8,), min_area=20
            )
            if blob.bbox[1] < 32
        ]
        print(
            "BRIDGE_RING_ABOARD", name, before_ring, after_ring,
            [
                (blob.bbox, blob.area)
                for blob in perception.connected_components(
                    branch.frame(), colors=(14,), min_area=2
                )
                if blob.bbox[1] < 40
            ],
            branch.levels_completed,
        )
    rotated_aboard = aboard.clone()
    rotated_aboard.step(*MAIN)
    print(
        "BRIDGE_RING_ABOARD_MAIN",
        [
            (blob.bbox, blob.area)
            for blob in perception.connected_components(
                rotated_aboard.frame(), colors=(14,), min_area=2
            )
            if blob.bbox[1] < 40
        ],
        rotated_aboard.levels_completed,
    )
    for action in (1, 2, 3, 4):
        branch = rotated_aboard.clone()
        branch.step(action)
        print(
            "BRIDGE_RING_ABOARD_MAIN_MOVE", action,
            [
                (blob.bbox, blob.area)
                for blob in perception.connected_components(
                    branch.frame(), colors=(14,), min_area=2
                )
                if blob.bbox[1] < 40
            ],
            branch.levels_completed,
        )


if __name__ == "__main__":
    arena.run_program("dc22", observe)

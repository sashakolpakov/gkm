"""Evaluate every cargo placement against every avatar movement region."""
from collections import deque
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

import perception
import solve
from probe_l6_ring_route import HUB_TO_BRIDGE
from probe_l6_right import (
    MAIN,
    SELECTOR,
    TOP,
    avatar_position,
    enter_right,
    movement_reach,
)


DIRECTIONS = {
    "U": (1, (6, 50, 34)),
    "D": (2, (6, 50, 40)),
    "L": (3, (6, 46, 36)),
    "R": (4, (6, 54, 36)),
}
TO_CENTER = {
    (56, 34): 2,
    (60, 34): 1,
    (58, 32): 4,
    (58, 36): 3,
}
CENTER = (58, 34)


def placement_key(env):
    return perception.arr(env.frame())[6:42, 14:34].tobytes()


def placement_label(env):
    return [
        (blob.bbox, blob.area)
        for blob in perception.connected_components(
            env.frame(), colors=(8,), min_area=20
        )
        if blob.bbox[1] < 32
    ]


def cargo_placements(root):
    queue = deque([root.clone()])
    seen = {placement_key(root)}
    out = [root.clone()]
    while queue:
        node = queue.popleft()
        position = avatar_position(node)
        for movement, control in DIRECTIONS.values():
            child = node.clone()
            if position != CENTER:
                child.step(TO_CENTER[position])
            child.step(movement)
            child.step(*control)
            child_key = placement_key(child)
            if child_key in seen:
                continue
            seen.add(child_key)
            out.append(child.clone())
            queue.append(child)
    return out


def won_reach(node):
    _, win = movement_reach(node)
    return win


def observe(env):
    solve.solve(env)
    base_level = env.levels_completed
    placements = cargo_placements(enter_right(env, 3))
    print("PLACEMENT_COUNT", len(placements))
    for index, placement in enumerate(placements):
        wins = {}
        wins["right3"] = won_reach(placement)
        hub = placement.clone()
        position = avatar_position(hub)
        if position != CENTER:
            hub.step(TO_CENTER[position])
        hub.step(*MAIN)
        if hub.levels_completed > base_level:
            wins["hub_arrival"] = ["control"]
        wins["hub"] = won_reach(hub)
        right0 = hub.clone()
        right0.step(*SELECTOR)
        right0.step(*MAIN)
        wins["right0"] = won_reach(right0)
        top = hub.clone()
        for _ in range(3):
            top.step(*SELECTOR)
        top.step(*MAIN)
        wins["top"] = won_reach(top)
        physical = hub.clone()
        for action in HUB_TO_BRIDGE:
            physical.step(action)
        physical.step(*TOP)
        physical.step(1)
        physical.step(3)
        physical.step(*TOP)
        wins["vertical"] = won_reach(physical)
        for _ in range(12):
            physical.step(1)
        physical.step(*MAIN)
        wins["horizontal"] = won_reach(physical)
        actual_wins = {
            name: path for name, path in wins.items() if path is not None
        }
        print(
            "PLACEMENT_EVAL", index, placement_label(placement),
            actual_wins,
        )
        if actual_wins:
            return


arena.run_program("dc22", observe)

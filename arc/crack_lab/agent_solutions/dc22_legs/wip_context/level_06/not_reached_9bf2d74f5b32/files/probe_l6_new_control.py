"""Characterize the control revealed after the lower transfer."""
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

import perception
import solve


TOP = (6, 54, 6)
MAIN = (6, 50, 26)
NEW = (6, 50, 46)
TO_BRIDGE = [3, 3, 3, 3, 3, 2, 2, 3, 3, 3, 2, 3]


def avatar_position(env):
    for blob in perception.connected_components(
            env.frame(), colors=(14,), min_area=2):
        if blob.bbox[1] < 32:
            return blob.top_left
    return None


def enter_upper(env):
    node = env.clone()
    for _ in range(4):
        node.step(*TOP)
    for action in TO_BRIDGE:
        node.step(action)
    node.step(*TOP)
    node.step(1)
    node.step(3)
    node.step(*TOP)
    for _ in range(6):
        node.step(1)
    node.step(*MAIN)
    for _ in range(14):
        node.step(1)
    return node


def pad(env):
    return tuple(
        int(value)
        for value in perception.arr(env.frame())[48:50, 18:20].flat
    )


def world_delta(before, after):
    delta = perception.frame_delta(before, after)
    samples = [sample for sample in delta["samples"] if sample[0] < 63]
    return delta["count"], delta["bbox"], samples


def observe(env):
    solve.solve(env)
    upper = enter_upper(env)
    cycle = upper.clone()
    print("NEW_CYCLE", 0, pad(cycle))
    for index in range(1, 9):
        before = perception.arr(cycle.frame()).copy()
        cycle.step(*NEW)
        print("NEW_CYCLE", index, pad(cycle), world_delta(before, cycle.frame()))
    for clicks in range(4):
        for side in ("left", "right"):
            node = upper.clone()
            for _ in range(clicks):
                node.step(*NEW)
            if side == "left":
                node.step(3)
            node.step(2)
            before_up = perception.arr(node.frame()).copy()
            node.step(1)
            print(
                "GLYPH_TEST", clicks, side, avatar_position(node),
                pad(node), world_delta(before_up, node.frame()),
                node.levels_completed,
            )
        north = upper.clone()
        for _ in range(clicks):
            north.step(*NEW)
        before_north = perception.arr(north.frame()).copy()
        north.step(1)
        print(
            "NORTH_TEST", clicks, avatar_position(north), pad(north),
            world_delta(before_north, north.frame()), north.levels_completed,
        )


arena.run_program("dc22", observe)

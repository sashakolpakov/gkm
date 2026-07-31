"""Check every static ring/rotator/bridge/selector configuration for the exit."""
from collections import deque
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

import perception
import solve
from probe_l6_right import (
    MAIN,
    SELECTOR,
    TOP,
    avatar_position,
    enter_right,
)


CENTER = (58, 34)
MOVES = (
    (1, (6, 50, 34)), (2, (6, 50, 40)),
    (3, (6, 46, 36)), (4, (6, 54, 36)),
)
TO_CENTER = {
    (56, 34): 2, (60, 34): 1, (58, 32): 4, (58, 36): 3,
}


def ring_key(env):
    return perception.arr(env.frame())[6:42, 6:34].tobytes()


def ring_label(env):
    return tuple(
        (blob.bbox, blob.area)
        for blob in perception.connected_components(
            env.frame(), colors=(8,), min_area=20
        )
        if blob.bbox[1] < 32
    )


def exit_tiles(env):
    return tuple(
        blob.bbox
        for blob in perception.connected_components(
            env.frame(), colors=(11,), min_area=4
        )
        if blob.area == 4
        and blob.size == (2, 2)
        and blob.bbox[1] < 40
    )


def placements(root):
    queue = deque([root.clone()])
    seen = {ring_key(root)}
    out = [root.clone()]
    while queue:
        node = queue.popleft()
        position = avatar_position(node)
        for movement, control in MOVES:
            child = node.clone()
            if position != CENTER:
                child.step(TO_CENTER[position])
            child.step(movement)
            child.step(*control)
            if avatar_position(child) != CENTER:
                child.step(TO_CENTER[avatar_position(child)])
            key = ring_key(child)
            if key in seen:
                continue
            seen.add(key)
            out.append(child.clone())
            queue.append(child)
    return out


def observe(env):
    solve.solve(env)
    base_level = env.levels_completed
    nodes = placements(enter_right(env, 3))
    print("STATIC_CONFIG_PLACEMENTS", len(nodes))
    checked = 0
    for index, placement in enumerate(nodes):
        orientations = [placement.clone()]
        alternate = placement.clone()
        alternate.step(1)
        alternate.step(*MAIN)
        alternate.step(2)
        orientations.append(alternate)
        for main_phase, oriented in enumerate(orientations):
            bridge = oriented.clone()
            for top_phase in range(6):
                selected = bridge.clone()
                for selector_offset in range(4):
                    tiles = exit_tiles(selected)
                    if tiles or selected.levels_completed > base_level:
                        print(
                            "STATIC_CONFIG_HIT", index, ring_label(selected),
                            main_phase, top_phase, selector_offset,
                            tiles, selected.levels_completed,
                        )
                        return
                    selected.step(*SELECTOR)
                    checked += 1
                bridge.step(*TOP)
        print("STATIC_CONFIG_DONE", index, checked)
    print("STATIC_CONFIG_NO_HIT", checked)


arena.run_program("dc22", observe)

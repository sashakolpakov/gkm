"""Probe affordances after reaching the upper-middle island."""
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

import perception
import solve


TOP = (6, 54, 6)
MAIN = (6, 50, 26)
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


def meaningful_delta(before, after):
    delta = perception.frame_delta(before, after)
    samples = [
        sample for sample in delta["samples"]
        if sample[:2] != (63, 0)
    ]
    return delta["count"], delta["bbox"], samples


def observe(env):
    solve.solve(env)
    upper = enter_upper(env)
    frame = perception.arr(upper.frame()).copy()
    print("UPPER", avatar_position(upper), upper.levels_completed)
    for y in range(2, 64, 4):
        for x in range(2, 64, 4):
            clone = upper.clone()
            clone.step(6, x, y)
            count, bbox, samples = meaningful_delta(frame, clone.frame())
            if samples:
                print("CLICK", (x, y), count, bbox, samples[:8])
    for start_col in (6, 8):
        node = upper.clone()
        while avatar_position(node)[1] > start_col:
            node.step(3)
        before = perception.arr(node.frame()).copy()
        node.step(2)
        print(
            "GLYPH", start_col, avatar_position(node),
            meaningful_delta(before, node.frame()),
            node.levels_completed,
        )


arena.run_program("dc22", observe)

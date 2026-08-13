"""Exhaust valid carrier hit pixels for level-9 transfer candidates."""

import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from perception import arr, frame_delta, safe_step


FIRST_RELAY = (
    ((42, 18), (42, 30)), ((48, 24), (36, 24)),
    ((42, 24), (30, 24)), ((36, 24), (24, 24)),
    ((30, 24), (18, 24)), ((18, 24), (18, 36)),
    ((18, 36), (30, 36)), ((24, 36), (36, 36)),
    ((30, 36), (42, 36)), ((36, 36), (48, 36)),
    ((48, 42), (48, 30)), ((48, 30), (36, 30)),
    ((48, 36), (36, 36)), ((36, 30), (36, 42)),
)


def change(before, after):
    left = arr(before).copy()
    right = arr(after).copy()
    right[0] = left[0]
    delta = frame_delta(left, right)
    return delta["count"], delta["bbox"]


def test_pixels(root, source, x_range, y_range):
    found = []
    for y in y_range:
        for x in x_range:
            child = root.clone()
            before = child.frame()
            safe_step(child, source)
            safe_step(child, (6, x, y))
            delta = change(before, child.frame())
            if delta[0] not in (0, 28):
                found.append(((x, y), delta, int(child.levels_completed)))
    return found


def probe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)
    for source, destination in FIRST_RELAY:
        safe_step(env, (6, source[1] + 1, source[0] + 1))
        safe_step(env, (6, destination[1] + 1, destination[0] + 1))

    print("bridge_pixels", test_pixels(
        env, (6, 17, 37), range(20, 34), range(34, 42)
    ))
    aligned = env.clone()
    for _ in range(7):
        safe_step(aligned, 4)
    print("incoming_pixels", test_pixels(
        aligned, (6, 11, 13), range(20, 28), range(34, 42)
    ))


arena.run_program("lf52", probe)

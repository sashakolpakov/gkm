"""Test explicit finish affordances after L9 has exactly one local peg."""

import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from perception import arr, connected_components, frame_delta, safe_step


EMPTY_OPENING = (
    ((42, 18), (42, 30)), ((48, 24), (36, 24)),
    ((42, 24), (30, 24)), ((36, 24), (24, 24)),
    ((30, 24), (18, 24)), ((18, 24), (18, 36)),
    ((18, 36), (30, 36)), ((24, 36), (36, 36)),
    ((30, 36), (42, 36)), ((36, 36), (48, 36)),
    ((48, 42), (48, 30)), ((48, 30), (36, 30)),
)


def move(node, source, destination):
    safe_step(node, (6, source[1] + 1, source[0] + 1))
    safe_step(node, (6, destination[1] + 1, destination[0] + 1))


def puzzle_delta(before, after):
    left = arr(before).copy()
    right = arr(after).copy()
    right[0] = left[0]
    result = frame_delta(left, right)
    return result["count"], result["bbox"]


def pieces(frame):
    return tuple(sorted(
        (blob.color, blob.bbox, blob.area)
        for blob in connected_components(
            frame, colors=(3, 7, 9, 11, 12, 14, 15)
        )
        if blob.color not in (9, 12) or blob.area >= 12
    ))


def probe(env):
    with open("checkpoint.json") as stream:
        campaign = json.load(stream)["final_path"]
    for action in campaign:
        safe_step(env, tuple(action) if isinstance(action, list) else action)
    for source, destination in EMPTY_OPENING:
        move(env, source, destination)

    base_level = int(env.levels_completed)
    baseline = env.frame()
    tests = {
        "reset": (7,),
        "reset_twice": (7, 7),
        "survivor_reset": ((6, 31, 37), 7),
        "survivor_arrow": ((6, 31, 37), (6, 60, 25)),
        "arrow_center": ((6, 60, 25),),
        "arrow_tip": ((6, 62, 27),),
        "arrow_then_reset": ((6, 60, 25), 7),
        "carrier_center": ((6, 43, 37),),
    }
    for name, actions in tests.items():
        node = env.clone()
        for action in actions:
            safe_step(node, action)
        print("finalize", name, int(node.levels_completed),
              puzzle_delta(baseline, node.frame()), pieces(node.frame()),
              flush=True)

    for action in (1, 2, 3, 4, 7):
        node = env.clone()
        changes = []
        for count in range(1, 13):
            before = node.frame()
            safe_step(node, action)
            delta = puzzle_delta(before, node.frame())
            if delta[0] or int(node.levels_completed) > base_level:
                changes.append((count, int(node.levels_completed), delta))
            if int(node.levels_completed) > base_level:
                break
        print("repeat", action, tuple(changes), flush=True)


if __name__ == "__main__":
    arena.run_program("lf52", probe)

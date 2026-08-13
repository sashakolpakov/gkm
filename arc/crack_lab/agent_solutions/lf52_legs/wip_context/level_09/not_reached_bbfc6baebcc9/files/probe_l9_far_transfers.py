"""Verify bridge transfers from the far board onto the carrier rail."""

import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from perception import arr, connected_components, frame_delta, safe_step


FIRST_RELAY = (
    ((42, 18), (42, 30)), ((48, 24), (36, 24)),
    ((42, 24), (30, 24)), ((36, 24), (24, 24)),
    ((30, 24), (18, 24)), ((18, 24), (18, 36)),
    ((18, 36), (30, 36)), ((24, 36), (36, 36)),
    ((30, 36), (42, 36)), ((36, 36), (48, 36)),
    ((48, 42), (48, 30)), ((48, 30), (36, 30)),
    ((48, 36), (36, 36)), ((36, 30), (36, 42)),
)


def compact(frame):
    return tuple(
        (blob.color, blob.bbox, blob.area)
        for blob in connected_components(frame, colors=(3, 9, 11, 12, 14, 15))
        if blob.color != 9 or blob.area >= 12
    )


def delta(before, after):
    left = arr(before).copy()
    right = arr(after).copy()
    right[0] = left[0]
    out = frame_delta(left, right)
    return out["count"], out["bbox"]


def pair(node, source, destination):
    child = node.clone()
    before = child.frame()
    safe_step(child, (6, source[1] + 1, source[0] + 1))
    safe_step(child, (6, destination[1] + 1, destination[0] + 1))
    return delta(before, child.frame()), int(child.levels_completed), compact(child.frame())


def probe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)
    for source, destination in FIRST_RELAY:
        safe_step(env, (6, source[1] + 1, source[0] + 1))
        safe_step(env, (6, destination[1] + 1, destination[0] + 1))

    for _ in range(9):
        safe_step(env, 4)
    for offset in range(9, 15):
        bridges = tuple(sorted(
            blob.top_left
            for blob in connected_components(env.frame(), colors=(9,))
            if blob.size == (4, 4) and blob.area == 12
        ))
        tests = []
        for source in bridges:
            for destination in (
                (30, source[1]), (6, source[1]),
                (36, 22), (30, 22), (12, source[1]),
            ):
                if destination == source:
                    continue
                tests.append((source, destination,
                              pair(env, source, destination)))
        print("offset", offset, tuple(tests), flush=True)
        safe_step(env, 4)


arena.run_program("lf52", probe)

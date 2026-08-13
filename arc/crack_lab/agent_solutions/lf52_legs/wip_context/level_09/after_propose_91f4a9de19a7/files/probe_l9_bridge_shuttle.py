"""Shuttle the far movable-bridge pair back to the isolated peg."""

import json
import os
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


def move(env, source, destination):
    safe_step(env, (6, source[1] + 1, source[0] + 1))
    safe_step(env, (6, destination[1] + 1, destination[0] + 1))


def compact(frame):
    return tuple(
        (blob.color, blob.bbox, blob.area)
        for blob in connected_components(frame, colors=(3, 9, 11, 12, 14, 15))
        if blob.color != 9 or blob.area >= 12
    )


def puzzle_delta(before, after):
    left = arr(before).copy()
    right = arr(after).copy()
    right[0] = left[0]
    out = frame_delta(left, right)
    return out["count"], out["bbox"]


def cells(frame):
    blobs = connected_components(frame, colors=(1, 9, 12, 14, 15))
    destinations = {
        blob.top_left for blob in blobs
        if blob.color == 1 and blob.size == (4, 4) and blob.area == 16
    }
    sources = {
        blob.top_left for blob in blobs
        if ((blob.color == 9 and blob.size == (4, 4) and blob.area == 12)
            or (blob.color == 14 and blob.size == (4, 4)))
    }
    destinations |= sources
    destinations |= {
        (blob.bbox[0] + 1, blob.bbox[1]) for blob in blobs
        if blob.color == 15 and blob.size == (4, 4) and blob.area == 12
    }
    destinations |= {
        blob.top_left for blob in blobs
        if blob.color == 12 and blob.size == (4, 4)
    }
    return tuple(sorted(sources)), tuple(sorted(destinations))


def actual_moves(root):
    sources, destinations = cells(root.frame())
    before = compact(root.frame())
    found = []
    for source in sources:
        for destination in destinations:
            if source == destination:
                continue
            child = root.clone()
            move(child, source, destination)
            after = compact(child.frame())
            if after != before or child.levels_completed > root.levels_completed:
                found.append((source, destination, int(child.levels_completed), after))
    return tuple(found)


def probe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)
    env = env.clone()
    for source, destination in FIRST_RELAY:
        move(env, source, destination)
    for _ in range(9):
        safe_step(env, 4)

    for source, destination in (
        ((18, 58), (18, 46)),
        ((18, 52), (18, 40)),
        ((18, 46), (18, 34)),
        ((18, 40), (18, 28)),
        ((18, 34), (18, 22)),
        ((18, 28), (18, 16)),
        ((18, 22), (18, 10)),
        ((18, 16), (18, 4)),
    ):
        move(env, source, destination)
    safe_step(env, 3)
    move(env, (18, 16), (18, 4))
    print("staged", int(env.levels_completed), compact(env.frame()), flush=True)

    before = env.frame()
    move(env, (12, 4), (24, 4))
    print("peg_hop", puzzle_delta(before, env.frame()),
          int(env.levels_completed), compact(env.frame()), flush=True)
    for source, destination in (
        ((18, 4), (30, 4)),
        ((24, 4), (36, 4)),
        ((18, 10), (30, 10)),
        ((36, 22), (24, 22)),
    ):
        child = env.clone()
        test_before = child.frame()
        move(child, source, destination)
        print("junction", source, destination,
              puzzle_delta(test_before, child.frame()),
              int(child.levels_completed), compact(child.frame()), flush=True)

    # Traverse the two fixed-bridge lanes, using the movable pair to switch
    # rows wherever a fixed chain ends.
    for source, destination in (
        ((24, 4), (24, 16)),
        ((18, 4), (18, 16)),
        ((24, 16), (12, 16)),
        ((12, 16), (12, 28)),
        ((12, 28), (12, 40)),
        ((18, 10), (18, 22)),
        ((18, 16), (18, 28)),
        ((18, 22), (18, 34)),
        ((18, 28), (18, 40)),
        ((12, 40), (24, 40)),
        ((24, 40), (24, 52)),
    ):
        move(env, source, destination)
    safe_step(env, 4)
    move(env, (24, 46), (24, 58))
    for source, destination in (
        ((18, 28), (18, 40)),
        ((18, 34), (18, 46)),
        ((18, 40), (18, 52)),
        ((18, 46), (18, 58)),
    ):
        move(env, source, destination)
    print("far_lane", int(env.levels_completed), compact(env.frame()), flush=True)

    before_transfer = env.frame()
    move(env, (18, 58), (30, 58))
    move(env, (24, 58), (36, 58))
    print("lower_transfer", puzzle_delta(before_transfer, env.frame()),
          int(env.levels_completed), compact(env.frame()), flush=True)
    for _ in range(5):
        safe_step(env, 4)
    before_capture = env.frame()
    move(env, (36, 22), (36, 34))
    print("final_capture", puzzle_delta(before_capture, env.frame()),
          int(env.levels_completed), compact(env.frame()), flush=True)
    if os.environ.get("L9_ACTUAL") == "1":
        print("actual_moves", actual_moves(env), flush=True)


arena.run_program("lf52", probe)

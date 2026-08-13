"""Compact observation of the level-9 board revealed after phase one."""

import json
import sys

sys.path.insert(0, "/private/tmp/gkm-submitted-protocol-lf52-20260805/arc/crack_lab")

import gkm_arena as arena

import numpy as np

from perception import action_deltas, arr, color_counts, connected_components, safe_step


PHASE_ONE = (
    (6, 19, 43), (6, 31, 43), (6, 25, 49), (6, 25, 37),
    (6, 25, 43), (6, 25, 31), (6, 25, 37), (6, 25, 25),
    (6, 25, 31), (6, 25, 19), (6, 25, 19), (6, 37, 19),
    (6, 37, 19), (6, 37, 31), (6, 37, 25), (6, 37, 37),
    (6, 37, 31), (6, 37, 43), (6, 37, 37), (6, 37, 49),
    (6, 43, 49), (6, 31, 49), (6, 31, 49), (6, 31, 37),
    (6, 37, 49), (6, 37, 37), (6, 31, 37), (6, 43, 37),
)


def state(frame):
    blobs = connected_components(frame, colors=(1, 9, 11, 14, 15))
    holes = tuple(
        blob.top_left for blob in blobs
        if blob.color == 1 and blob.size == (4, 4) and blob.area == 16
    )
    pegs = tuple(
        blob.top_left for blob in blobs
        if blob.color == 14 and blob.size == (4, 4)
    )
    bridges = tuple(
        blob.top_left for blob in blobs
        if blob.color == 9 and blob.size == (4, 4) and blob.area == 12
    )
    persistent = tuple(
        (blob.bbox[0] + 1, blob.bbox[1]) for blob in blobs
        if blob.color == 15 and blob.size == (4, 4)
    )
    carriers = tuple(
        (blob.bbox[0] + 1, blob.bbox[1] + 1) for blob in blobs
        if blob.color == 11 and blob.area >= 4
    )
    return holes, pegs, bridges, persistent, carriers


def world_key(frame):
    return arr(frame)[1:, :].tobytes()


def advance(root, path):
    node = root.clone()
    for action in path:
        safe_step(node, action)
    return node


def legal_moves(root):
    root_key = world_key(root.frame())
    _, pegs, bridges, _, _ = state(root.frame())
    found = []
    for row, col in pegs + bridges:
        source = (6, col + 1, row + 1)
        if not (0 <= source[1] < 64 and 0 <= source[2] < 64):
            continue
        picked = advance(root, (source,))
        if world_key(picked.frame()) == root_key:
            continue
        for dr, dc in ((-12, 0), (12, 0), (0, -12), (0, 12)):
            destination = (6, col + dc + 1, row + dr + 1)
            if not (0 <= destination[1] < 64 and 0 <= destination[2] < 64):
                continue
            child = advance(root, (source, destination))
            child_frame = arr(child.frame())[1:, :]
            if (
                world_key(child.frame()) != root_key
                and not np.any(child_frame == 2)
                and not np.any(child_frame == 3)
            ):
                found.append(((row, col), (row + dr, col + dc), state(child.frame())))
    return tuple(found)


def probe(env):
    with open("checkpoint.json") as handle:
        checkpoint = json.load(handle)
    for action in checkpoint["final_path"]:
        env.step(action)
    root = env.clone()
    for action in PHASE_ONE:
        safe_step(root, action)

    frame = root.frame()
    blobs = tuple(
        (blob.color, blob.bbox, blob.area)
        for blob in connected_components(frame, min_area=2)
        if blob.color != 10
    )
    deltas = {
        action: (delta["count"], delta["bbox"])
        for action, delta in action_deltas(root).items()
    }
    print("PHASE2_STATE", root.levels_completed, tuple(root.actions), color_counts(frame))
    print("PHASE2_BLOBS", blobs)
    print("PHASE2_DELTAS", deltas)
    print("PHASE2_SYMBOLIC", state(frame))
    print("PHASE2_LEGAL", legal_moves(root))

    for action in (1, 2, 3, 4):
        node = root.clone()
        observations = []
        for count in range(1, 13):
            before = world_key(node.frame())
            safe_step(node, action)
            changed = sum(a != b for a, b in zip(before, world_key(node.frame())))
            if changed:
                observations.append((count, changed, state(node.frame())))
        print("PHASE2_RUN", action, tuple(observations), "level", node.levels_completed)

    for path in ((4,), (4, 3), (4, 4), (4, 4, 1), (4, 4, 2), (4, 4, 3)):
        child = advance(root, path)
        print(
            "PHASE2_PATH", path, "level", child.levels_completed,
            "state", state(child.frame()), "legal", legal_moves(child),
        )

    unload = ((6, 23, 37), (6, 11, 37))
    unloaded = advance(root, unload)
    print("PHASE2_UNLOAD", state(unloaded.frame()), "legal", legal_moves(unloaded))
    for path in (
        unload + (1,), unload + (2,), unload + (3,), unload + (4,),
        unload + (4, 4), unload + (4, 1), unload + (4, 2),
    ):
        child = advance(root, path)
        print(
            "PHASE2_EMPTY_CARRIER", path[len(unload):], "level", child.levels_completed,
            "state", state(child.frame()), "legal", legal_moves(child),
        )

    reload_once = unload + ((6, 11, 37), (6, 23, 37))
    reloaded = advance(root, reload_once)
    print("PHASE3_RELOAD", state(reloaded.frame()), "legal", legal_moves(reloaded))
    for path in (
        reload_once + (1,), reload_once + (2,), reload_once + (3,), reload_once + (4,),
        reload_once + (4, 4), reload_once + (4, 4, 4), reload_once + (3, 3),
    ):
        child = advance(root, path)
        print(
            "PHASE3_PATH", path[len(reload_once):], "level", child.levels_completed,
            "state", state(child.frame()), "legal", legal_moves(child),
        )

    node = reloaded.clone()
    for shift in range(16):
        print(
            "PHASE3_SCAN", shift, "level", node.levels_completed,
            "state", state(node.frame())[1:], "legal", legal_moves(node),
        )
        before = world_key(node.frame())
        safe_step(node, 4)
        if world_key(node.frame()) == before:
            break


levels, path, error = arena.run_program("lf52", probe)
print("PROBE_RESULT", levels, len(path), error)

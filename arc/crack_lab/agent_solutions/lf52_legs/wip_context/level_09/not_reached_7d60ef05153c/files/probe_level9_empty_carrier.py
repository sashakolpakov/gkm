"""Bounded symbolic probes for the empty carrier at pristine level 9."""

import json

import gkm_try

from perception import arr, connected_components, safe_step


def compact(frame):
    blobs = connected_components(frame, colors=(9, 11, 12, 14, 15))
    carriers = {
        blob.top_left
        for blob in blobs
        if blob.color == 12 and blob.size == (4, 4)
    } | {
        (blob.bbox[0] + 1, blob.bbox[1] + 1)
        for blob in blobs
        if blob.color == 11 and blob.area >= 4
    }
    return (
        tuple(sorted(carriers)),
        tuple(sorted(blob.top_left for blob in blobs
                     if blob.color == 14 and blob.size == (4, 4))),
        tuple(sorted(blob.top_left for blob in blobs
                     if blob.color == 9 and blob.size == (4, 4))),
        tuple(sorted((blob.bbox[0] + 1, blob.bbox[1]) for blob in blobs
                     if blob.color == 15 and blob.size == (4, 4))),
    )


def label(path):
    names = {1: "U", 2: "D", 3: "L", 4: "R"}
    return "".join(names[action] for action in path) or "-"


def probe(env):
    with open("checkpoint.json") as checkpoint_file:
        prefix = json.load(checkpoint_file)["final_path"]
    for action in prefix:
        safe_step(env, action)

    base_level = int(env.levels_completed)
    base = compact(env.frame())
    observations = {}
    rewards = []

    routes = []
    for action in (1, 2, 3, 4):
        for count in range(1, 13):
            routes.append((action,) * count)
    for vertical in (1, 2):
        for count in range(1, 13):
            routes.append((4,) * 4 + (vertical,) * count + (3,))
    for extra_right in range(1, 9):
        for left in range(1, 13):
            routes.append((4,) * (4 + extra_right) + (3,) * left)

    for path in routes:
        node = env.clone()
        for action in path:
            safe_step(node, action)
        signature = compact(node.frame())
        previous = observations.get(signature)
        if previous is None or len(path) < len(previous):
            observations[signature] = path
        if int(node.levels_completed) != base_level or signature[3]:
            rewards.append((label(path), int(node.levels_completed), signature))

    ordered = sorted(
        ((len(path), label(path), signature) for signature, path in observations.items()),
        key=lambda item: (item[0], item[1]),
    )
    print("EMPTY_BASE", base_level, base)
    print("EMPTY_ROUTES", len(routes), "UNIQUE", len(ordered))
    for item in ordered:
        print("EMPTY_STATE", item)
    print("EMPTY_PHASE_OR_REWARD", rewards)


gkm_try.A.run_program("lf52", probe)

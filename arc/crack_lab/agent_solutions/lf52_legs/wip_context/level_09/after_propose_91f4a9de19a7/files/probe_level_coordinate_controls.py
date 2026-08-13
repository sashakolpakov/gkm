"""Test non-piece components as coordinate controls along a known route."""

import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from perception import arr, connected_components, safe_step


def normalize(action):
    return tuple(action) if isinstance(action, list) else action


def split(path):
    groups = []
    keys = []
    index = 0
    while index < len(path):
        action = normalize(path[index])
        if isinstance(action, int):
            keys.append(action)
            index += 1
        else:
            groups.append((tuple(keys), (action, normalize(path[index + 1]))))
            keys = []
            index += 2
    return tuple(groups)


def key(node):
    return int(node.levels_completed), arr(node.frame())[1:, :].tobytes()


def pieces(frame):
    return tuple(sorted(
        (blob.color, blob.bbox, blob.area)
        for blob in connected_components(frame, colors=(3, 8, 9, 12, 14))
        if blob.area >= 4
    ))


def probe(env):
    desired = int(os.environ.get("OPT_LEVEL", "7"))
    first_stage = int(os.environ.get("OPT_FIRST_STAGE", "1"))
    last_stage = int(os.environ.get("OPT_LAST_STAGE", "999"))
    with open("checkpoint.json") as stream:
        campaign = tuple(normalize(action)
                         for action in json.load(stream)["final_path"])

    prior = int(env.levels_completed)
    entry = None
    start = end = None
    for index, action in enumerate(campaign):
        safe_step(env, action)
        current = int(env.levels_completed)
        if prior < desired - 1 <= current:
            entry = env.clone()
            start = index + 1
        if prior < desired <= current:
            end = index + 1
            break
        prior = current

    node = entry
    for stage, (keys, clicks) in enumerate(split(campaign[start:end])):
        for action in keys:
            safe_step(node, action)
        if first_stage <= stage <= last_stage:
            baseline = {}
            for action in (1, 2, 3, 4):
                child = node.clone()
                safe_step(child, action)
                baseline[action] = key(child)
            colors = tuple(sorted(set(int(value)
                                      for value in arr(node.frame())[1:].flat)))
            blobs = [
                blob for blob in connected_components(node.frame(), colors=colors)
                if blob.color not in (1, 3, 8, 12, 14, 15)
                and blob.area <= 400
            ]
            hits = []
            for blob in blobs:
                top, left = blob.top_left
                row = min(63, top + 1)
                col = min(63, left + 1)
                clicked = node.clone()
                safe_step(clicked, (6, col, row))
                direct = key(clicked) != key(node)
                changed_keys = []
                details = []
                for action in (1, 2, 3, 4):
                    child = clicked.clone()
                    safe_step(child, action)
                    if key(child) != baseline[action]:
                        changed_keys.append(action)
                        if os.environ.get("OPT_VERBOSE") == "1":
                            reference = node.clone()
                            safe_step(reference, action)
                            details.append((action, pieces(reference.frame()),
                                            pieces(child.frame())))
                if direct or changed_keys:
                    hits.append((blob.color, blob.bbox, blob.area,
                                 (col, row), direct, tuple(changed_keys),
                                 tuple(details)))
            print("controls", desired, stage, "tested", len(blobs),
                  "hits", tuple(hits), flush=True)
        for action in clicks:
            safe_step(node, action)


arena.run_program("lf52", probe)

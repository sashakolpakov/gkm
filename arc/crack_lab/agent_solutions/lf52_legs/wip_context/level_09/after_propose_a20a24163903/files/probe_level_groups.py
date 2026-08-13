"""Small symbolic group trace for one checkpoint level."""

import json
import os

import gkm_try

from perception import connected_components, safe_step


LEVEL_ENDS = {1: 8, 2: 42, 3: 87, 4: 149, 5: 238, 6: 331, 7: 476, 8: 544}
TARGET = int(os.environ.get("TARGET_LEVEL", "8"))


def compact(frame):
    blobs = connected_components(frame, colors=(1, 8, 9, 11, 12, 14, 15))
    carriers = {
        blob.top_left for blob in blobs
        if blob.color == 12 and blob.size == (4, 4)
    } | {
        (blob.bbox[0] + 1, blob.bbox[1] + 1) for blob in blobs
        if blob.color == 11 and blob.area >= 4
    }
    return {
        "C": tuple(sorted(carriers)),
        "B8": tuple(sorted(blob.top_left for blob in blobs
                           if blob.color == 8 and blob.size == (4, 4))),
        "B9": tuple(sorted(blob.top_left for blob in blobs
                           if blob.color == 9 and blob.size == (4, 4))),
        "P": tuple(sorted(blob.top_left for blob in blobs
                          if blob.color == 14 and blob.size == (4, 4))),
        "F": tuple(sorted((blob.bbox[0] + 1, blob.bbox[1]) for blob in blobs
                          if blob.color == 15 and blob.size == (4, 4))),
    }


def is_click(action):
    return isinstance(action, (list, tuple))


def probe(env):
    with open("checkpoint.json") as checkpoint_file:
        path = json.load(checkpoint_file)["final_path"]
    start, end = LEVEL_ENDS[TARGET - 1], LEVEL_ENDS[TARGET]
    for action in path[:start]:
        safe_step(env, action)
    print("GROUP_ENTRY", TARGET, compact(env.frame()))
    index = start
    group = 0
    while index < end:
        keys = []
        while index < end and not is_click(path[index]):
            keys.append(path[index]); safe_step(env, path[index]); index += 1
        before = compact(env.frame())
        clicks = []
        while index < end and is_click(path[index]) and len(clicks) < 2:
            action = tuple(path[index]); clicks.append(action)
            safe_step(env, action); index += 1
        print(
            "GROUP", group, "K", "".join(map(str, keys)), "M", tuple(clicks),
            "BEFORE", before, "AFTER", compact(env.frame()),
            "L", int(env.levels_completed),
        )
        group += 1


gkm_try.A.run_program("lf52", probe)

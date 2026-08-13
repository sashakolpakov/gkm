"""Describe one verified coordinate pair at a chosen route stage."""

import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from legs import _bridge_carrier_moves
from perception import arr, connected_components, frame_delta, safe_step


def normalize(action):
    return tuple(action) if isinstance(action, list) else action


def split(path):
    groups = []
    keys = []
    index = 0
    while index < len(path):
        if isinstance(path[index], int):
            keys.append(path[index])
            index += 1
        else:
            groups.append((tuple(keys), (path[index], path[index + 1])))
            keys = []
            index += 2
    return tuple(groups)


def compact(frame):
    return tuple(
        (blob.color, blob.bbox, blob.area)
        for blob in connected_components(
            frame, colors=(1, 3, 8, 9, 11, 12, 14, 15)
        )
        if blob.area >= 4 and (
            blob.color not in (1, 9, 11) or blob.size[0] <= 6
        )
    )


def delta(before, after):
    left = arr(before).copy()
    right = arr(after).copy()
    right[0] = left[0]
    out = frame_delta(left, right)
    return out["count"], out["bbox"], tuple(out.get("samples", ())[:12])


def local(frame, clicks):
    image = arr(frame)
    first, second = clicks
    source = (first[2] - 1, first[1] - 1)
    destination = (second[2] - 1, second[1] - 1)
    midpoint = ((source[0] + destination[0]) // 2,
                (source[1] + destination[1]) // 2)
    return tuple(
        (point, tuple(tuple(int(value) for value in row)
                      for row in image[point[0]:point[0] + 4,
                                       point[1]:point[1] + 4]))
        for point in (source, midpoint, destination)
    )


def probe(env):
    desired = int(os.environ.get("OPT_LEVEL", "7"))
    stage = int(os.environ.get("OPT_STAGE", "9"))
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
    groups = split(campaign[start:end])
    node = entry
    for keys, clicks in groups[:stage]:
        for action in keys + clicks:
            safe_step(node, action)
    keys, clicks = groups[stage]
    source_text = os.environ.get("OPT_SOURCE")
    destination_text = os.environ.get("OPT_DESTINATION")
    if source_text and destination_text:
        source = tuple(int(value) for value in source_text.split(","))
        destination = tuple(
            int(value) for value in destination_text.split(",")
        )
        clicks = (
            (6, source[1] + 1, source[0] + 1),
            (6, destination[1] + 1, destination[0] + 1),
        )
    for action in keys:
        safe_step(node, action)
    before = node.frame()
    safe_step(node, clicks[0])
    selected = node.frame()
    safe_step(node, clicks[1])
    after = node.frame()
    print("click_effect", desired, stage, keys, clicks,
          "local", local(before, clicks),
          "first", delta(before, selected),
          "second", delta(selected, after),
          "moves_after_first", _bridge_carrier_moves(selected),
          "objects_before", compact(before),
          "objects_after", compact(after), flush=True)


arena.run_program("lf52", probe)

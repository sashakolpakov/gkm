"""Detect whether clicking visible components changes subsequent key control."""

import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from perception import arr, connected_components, frame_delta, safe_step


def normalize(action):
    return tuple(action) if isinstance(action, list) else action


def delta(left_frame, right_frame):
    left = arr(left_frame).copy()
    right = arr(right_frame).copy()
    right[0] = left[0]
    result = frame_delta(left, right)
    return result["count"], result["bbox"]


def probe(env):
    desired = int(os.environ.get("OPT_LEVEL", "7"))
    with open("checkpoint.json") as stream:
        campaign = tuple(normalize(action)
                         for action in json.load(stream)["final_path"])
    prior = int(env.levels_completed)
    entry = None
    for action in campaign:
        safe_step(env, action)
        current = int(env.levels_completed)
        if prior < desired - 1 <= current:
            entry = env.clone()
            break
        prior = current

    prefix_text = os.environ.get("OPT_KEYS", "")
    if prefix_text:
        for action in tuple(int(value) for value in prefix_text.split(",")):
            safe_step(entry, action)

    base_frame = entry.frame()
    colors = tuple(sorted(set(int(value) for value in arr(base_frame)[1:].flat)))
    blobs = connected_components(base_frame, colors=colors)
    if os.environ.get("OPT_PIECES") == "1":
        blobs = [blob for blob in blobs
                 if blob.color in (8, 12, 14, 15)
                 and blob.size in ((2, 2), (4, 4))]
    baseline = {}
    for action in (1, 2, 3, 4, 7):
        node = entry.clone()
        safe_step(node, action)
        baseline[action] = node.frame()

    changed = []
    tested = 0
    for index, blob in enumerate(blobs):
        top, left = blob.top_left
        row, col = min(63, top + 1), min(63, left + 1)
        if not (0 <= row <= 63 and 0 <= col <= 63):
            continue
        tested += 1
        clicked = entry.clone()
        safe_step(clicked, (6, col, row))
        click_delta = delta(base_frame, clicked.frame())
        key_deltas = []
        for action in (1, 2, 3, 4, 7):
            child = clicked.clone()
            safe_step(child, action)
            key_deltas.append((action,) + delta(baseline[action], child.frame()))
        if click_delta[0] or any(item[1] for item in key_deltas):
            changed.append((index, blob.color, blob.bbox, blob.area,
                            (col, row), click_delta, tuple(key_deltas)))
    print("components", tested, "colors", colors, "changed", len(changed),
          flush=True)
    for item in changed:
        print("component", item, flush=True)


arena.run_program("lf52", probe)

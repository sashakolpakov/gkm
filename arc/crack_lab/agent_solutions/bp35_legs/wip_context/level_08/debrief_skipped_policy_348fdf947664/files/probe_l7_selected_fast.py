"""Replay BFS with an explicit selected-object history token."""

from collections import deque
import hashlib
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from legs import (
    COL_ANCHORS, ROW_ANCHORS, _cell_shape, click_action, run_actions,
)
from perception import connected_components
from probe_l7_frontier import BASE
from probe_l7_raw_search import avatar_position, target_path_distance


def selection_token(frame, action):
    digest = hashlib.blake2b(
        np.asarray(frame)[:63].tobytes(), digest_size=8
    ).digest()
    return digest, action


def controls(frame, ay):
    visible = [
        blob
        for blob in connected_components(frame, colors=(8,), min_area=3)
        if blob.bbox[1] <= 5 and blob.bbox[0] < 63
    ]
    if not visible:
        return []
    blob = min(visible, key=lambda item: abs(item.centroid[0] - ay))
    y, x = blob.centroid
    return [(6, int(round(x)), int(round(y)))]


def choices(frame):
    avatar = avatar_position(frame)
    if avatar is None:
        return []
    ax, ay = avatar
    ai = min(range(10), key=lambda i: abs(ROW_ANCHORS[i] - ay))
    aj = min(range(8), key=lambda j: abs(COL_ANCHORS[j] - ax))
    out = [(3,), (4,), (7,), *controls(frame, ay)]
    for i in range(max(0, ai - 3), min(10, ai + 4)):
        for j in range(max(0, aj - 1), min(8, aj + 2)):
            if _cell_shape(frame, i, j)[0] in (12, 14):
                out.append(click_action(i, j))
    return list(dict.fromkeys(out))


def probe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)
    run_actions(env, BASE)
    base_level = int(env.levels_completed)
    root = env.clone()
    root_token = ("base-selected",)
    root_key = (np.asarray(root.frame())[:63].tobytes(), root_token)
    queue = deque([((), root_token)])
    seen = {root_key}
    expanded = 0
    best_distance = target_path_distance(root.frame())
    started = time.monotonic()
    max_states = int(os.environ.get("MAX_STATES", "500"))
    max_depth = int(os.environ.get("MAX_DEPTH", "28"))
    while queue and expanded < max_states:
        path, token = queue.popleft()
        node = root.clone()
        run_actions(node, path)
        expanded += 1
        if len(path) >= max_depth or node.terminal():
            continue
        before = np.asarray(node.frame()).copy()
        for action in choices(before):
            child = node.clone()
            child.step(*action)
            child_path = (*path, action)
            if child.levels_completed > base_level:
                route = [*BASE, *child_path]
                print(
                    "SELECTED_FAST_WIN", expanded, len(route), route,
                    flush=True,
                )
                return
            if child.terminal() or avatar_position(child.frame()) is None:
                continue
            child_token = (
                selection_token(before, action)
                if action[0] == 6 else token
            )
            child_frame = np.asarray(child.frame())
            key = (child_frame[:63].tobytes(), child_token)
            if key in seen:
                continue
            seen.add(key)
            queue.append((child_path, child_token))
            distance = target_path_distance(child_frame)
            if distance is not None and (
                best_distance is None or distance < best_distance
            ):
                best_distance = distance
                print(
                    "SELECTED_FAST_TARGET", distance, expanded,
                    len(child_path), child_path, flush=True,
                )
        if expanded % 50 == 0:
            print(
                "SELECTED_FAST_SEARCH", expanded, len(queue), len(seen),
                len(path), best_distance,
                round(time.monotonic() - started, 1), flush=True,
            )
    print(
        "SELECTED_FAST_DONE", expanded, len(queue), len(seen),
        best_distance, flush=True,
    )


levels, path, error = arena.run_program("bp35", probe)
print("RESULT", levels, len(path), error)

"""Find an exact shorter undo-assisted carrier alignment."""

from collections import deque
import json
import os

import gkm_try

from perception import arr, safe_step
from probe_undo_slide import LEVEL_ENDS, frame_key, groups


TARGET_LEVEL = int(os.environ.get("TARGET_LEVEL", "7"))
TARGET_GROUP = int(os.environ.get("TARGET_GROUP", "9"))
MAX_STATES = int(os.environ.get("MAX_STATES", "800"))
MAX_KEY_DEPTH = int(os.environ.get("MAX_KEY_DEPTH", "9"))


def probe(env):
    with open("checkpoint.json") as checkpoint_file:
        full = json.load(checkpoint_file)["final_path"]
    start, end = LEVEL_ENDS[TARGET_LEVEL - 1], LEVEL_ENDS[TARGET_LEVEL]
    for action in full[:start]:
        safe_step(env, action)
    level_groups = groups(full[start:end])
    for keys, pair in level_groups[:TARGET_GROUP]:
        for action in keys + pair:
            safe_step(env, action)
    root = env.clone(); keys, pair = level_groups[TARGET_GROUP]
    baseline = root.clone()
    for action in keys:
        safe_step(baseline, action)
    target = frame_key(baseline)
    redo = level_groups[TARGET_GROUP - 1][1][-1]

    undone = root.clone(); safe_step(undone, 7)
    queue = deque([(undone, ())]); seen = {frame_key(undone)}; found = None
    while queue and len(seen) <= MAX_STATES:
        node, path = queue.popleft()
        replay = node.clone(); safe_step(replay, redo)
        if frame_key(replay) == target:
            found = path; break
        if len(path) >= MAX_KEY_DEPTH:
            continue
        for action in (1, 2, 3, 4):
            child = node.clone(); safe_step(child, action)
            key = frame_key(child)
            if key in seen:
                continue
            seen.add(key); queue.append((child, path + (action,)))
    print("UNDO_BFS", TARGET_LEVEL, TARGET_GROUP, len(keys), len(seen), found)
    if found is None:
        return
    candidate = root.clone()
    for action in (7,) + found + (redo,):
        safe_step(candidate, action)
    for action in pair:
        safe_step(candidate, action)
    for suffix_keys, suffix_pair in level_groups[TARGET_GROUP + 1:]:
        for action in suffix_keys + suffix_pair:
            safe_step(candidate, action)
        if int(candidate.levels_completed) >= TARGET_LEVEL:
            break
    print("UNDO_BFS_SUFFIX", 2 + len(found), int(candidate.levels_completed))


gkm_try.A.run_program("lf52", probe)

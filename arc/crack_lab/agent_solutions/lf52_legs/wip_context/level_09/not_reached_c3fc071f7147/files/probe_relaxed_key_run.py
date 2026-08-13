"""Shorten a key run by validating its coordinate move and full suffix."""

import json
import os
import sys
from collections import deque

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from perception import arr, safe_step


def normalize(action):
    return tuple(action) if isinstance(action, list) else action


def key(env):
    return int(env.levels_completed), arr(env.frame())[1:, :].tobytes()


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


def probe(env):
    desired = int(os.environ.get("OPT_LEVEL", "5"))
    choice = int(os.environ.get("OPT_GROUP", "3"))
    max_states = int(os.environ.get("OPT_STATES", "500"))
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
    root = entry.clone()
    for keys, clicks in groups[:choice]:
        for action in keys + clicks:
            safe_step(root, action)
    original_keys, clicks = groups[choice]
    suffix = tuple(
        action
        for keys, group_clicks in groups[choice + 1:]
        for action in keys + group_clicks
    )
    print("run", desired, choice, len(original_keys), clicks,
          len(suffix), flush=True)

    queue = deque([(root.clone(), ())])
    seen = {key(root)}
    tested = 0
    while queue and len(seen) <= max_states:
        node, path = queue.popleft()
        if len(path) < len(original_keys):
            moved = node.clone()
            before = key(moved)
            for action in clicks:
                safe_step(moved, action)
            if key(moved) != before:
                tested += 1
                result = moved.clone()
                for action in suffix:
                    safe_step(result, action)
                    if int(result.levels_completed) >= desired:
                        print("solution", len(path), path, tested,
                              len(seen), flush=True)
                        return
        if len(path) >= len(original_keys) - 1:
            continue
        for action in (1, 2, 3, 4):
            child = node.clone()
            safe_step(child, action)
            child_key = key(child)
            if child_key in seen:
                continue
            seen.add(child_key)
            queue.append((child, path + (action,)))
    print("none", len(seen), tested, flush=True)


arena.run_program("lf52", probe)

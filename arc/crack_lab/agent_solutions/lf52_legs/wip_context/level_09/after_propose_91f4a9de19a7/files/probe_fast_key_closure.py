"""Enumerate a route milestone's key graph using verified live undo."""

import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from perception import arr, safe_step
from probe_key_neighborhood_events import generic_moves


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


def frame_key(node):
    return int(node.levels_completed), arr(node.frame())[1:, :].tobytes()


def probe(env):
    desired = int(os.environ.get("OPT_LEVEL", "5"))
    stage = int(os.environ.get("OPT_STAGE", "4"))
    max_depth = int(os.environ.get("OPT_DEPTH", "16"))
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
    node = entry.clone()
    for keys, clicks in groups[:stage]:
        for action in keys + clicks:
            safe_step(node, action)

    lead_text = os.environ.get("OPT_LEAD", "")
    lead = tuple(int(value) for value in lead_text.split(",") if value)
    for action in lead:
        safe_step(node, action)

    root = frame_key(node)
    paths = {root: ()}
    events = {}
    steps = 0
    failed = None
    inverse = {1: 2, 2: 1, 3: 4, 4: 3}

    def visit(depth):
        nonlocal steps, failed
        state = frame_key(node)
        path = paths[state]
        for move in generic_moves(node.frame()):
            known = events.get(move)
            if known is None or len(path) < len(known):
                events[move] = path
        if depth >= max_depth or len(paths) >= max_states or failed:
            return
        order = tuple(int(value) for value in
                      os.environ.get("OPT_ORDER", "1,2,3,4").split(","))
        for action in order:
            before = frame_key(node)
            safe_step(node, action)
            steps += 1
            after = frame_key(node)
            if after == before:
                continue
            if after not in paths and len(paths) < max_states:
                paths[after] = path + (action,)
                visit(depth + 1)
            safe_step(node, inverse[action])
            steps += 1
            restored = frame_key(node)
            if restored != before:
                failed = (depth, action, path, restored[0], before[0])
                return

    visit(0)
    print("fast_closure", desired, stage, lead, len(paths), steps, failed,
          tuple(sorted((len(path), move, path)
                       for move, path in events.items())), flush=True)


arena.run_program("lf52", probe)

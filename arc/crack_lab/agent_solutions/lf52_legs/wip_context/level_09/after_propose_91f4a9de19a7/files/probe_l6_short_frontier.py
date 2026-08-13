"""Enumerate events after the nine-key alternate level-6 transfer."""

import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from perception import safe_step
from probe_level_event_closures import closure


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


def probe(env):
    with open("checkpoint.json") as stream:
        campaign = tuple(normalize(action)
                         for action in json.load(stream)["final_path"])
    prior = int(env.levels_completed)
    entry = None
    start = end = None
    for index, action in enumerate(campaign):
        safe_step(env, action)
        current = int(env.levels_completed)
        if prior < 5 <= current:
            entry = env.clone()
            start = index + 1
        if prior < 6 <= current:
            end = index + 1
            break
        prior = current
    groups = split(campaign[start:end])
    node = entry.clone()
    for keys, clicks in groups[:11]:
        for action in keys + clicks:
            safe_step(node, action)
    shortcut = (4,) * 7 + (1, 1)
    for action in shortcut + groups[11][1]:
        safe_step(node, action)
    if os.environ.get("OPT_FORWARD") == "1":
        for action in groups[12][1]:
            safe_step(node, action)
    states, events = closure(
        node,
        int(os.environ.get("OPT_STATES", "180")),
        int(os.environ.get("OPT_DEPTH", "20")),
    )
    print("l6_short_frontier", states,
          tuple(sorted((len(path), move, path)
                       for move, path in events.items())), flush=True)


arena.run_program("lf52", probe)

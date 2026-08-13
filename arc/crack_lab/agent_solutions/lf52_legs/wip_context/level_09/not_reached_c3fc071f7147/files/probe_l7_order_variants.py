"""Replay compact reorderings of the forced early level-7 handoffs."""

import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from perception import arr, safe_step


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
    return groups


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
        if prior < 6 <= current:
            entry = env.clone()
            start = index + 1
        if prior < 7 <= current:
            end = index + 1
            break
        prior = current
    groups = split(campaign[start:end])
    groups[21] = ((1, 1, 4, 4, 4, 2, 4, 2), groups[21][1])

    orders = {
        "early2": (0, 1, 4, 5, 2, 3) + tuple(range(6, len(groups))),
        "early3": (0, 1, 4, 5, 6, 2, 3) + tuple(range(7, len(groups))),
    }
    for name, order in orders.items():
        node = entry.clone()
        unchanged = []
        actions = 0
        for stage, group_index in enumerate(order):
            keys, clicks = groups[group_index]
            for action in keys:
                safe_step(node, action)
                actions += 1
            before = arr(node.frame())[1:, :].tobytes()
            for action in clicks:
                safe_step(node, action)
                actions += 1
            if arr(node.frame())[1:, :].tobytes() == before:
                unchanged.append((stage, group_index))
        print("order", name, actions, int(node.levels_completed),
              tuple(unchanged), flush=True)


arena.run_program("lf52", probe)

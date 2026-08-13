"""Replay equal-cost alternate controller states through verified suffixes."""

import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from perception import safe_step


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


def flatten(groups):
    return tuple(action for keys, clicks in groups for action in keys + clicks)


def entry_and_path(env, campaign, desired):
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
    return entry, campaign[start:end]


def run(entry, desired, name, groups):
    node = entry.clone()
    path = flatten(groups)
    completed = None
    for index, action in enumerate(path, 1):
        safe_step(node, action)
        if int(node.levels_completed) >= desired:
            completed = index
            break
    print("alternate", desired, name, len(path), completed,
          int(node.levels_completed), flush=True)


def probe(env):
    with open("checkpoint.json") as stream:
        campaign = tuple(normalize(action)
                         for action in json.load(stream)["final_path"])

    entry5, path5 = entry_and_path(env.clone(), campaign, 5)
    base5 = split(path5)
    for name, changes in (
        ("stage2", {2: (4, 4, 4, 2, 7, 2)}),
        ("stage3", {3: (2, 3, 2, 2, 4, 4, 4, 1, 1, 1, 1,
                        4, 4, 4, 2)}),
        ("both", {2: (4, 4, 4, 2, 7, 2),
                  3: (2, 3, 2, 2, 4, 4, 4, 1, 1, 1, 1,
                      4, 4, 4, 2)}),
    ):
        groups = list(base5)
        for stage, keys in changes.items():
            groups[stage] = (keys, groups[stage][1])
        run(entry5, 5, name, groups)

    entry7, path7 = entry_and_path(env.clone(), campaign, 7)
    base7 = split(path7)
    base7[21] = ((1, 1, 4, 4, 4, 2, 4, 2), base7[21][1])
    for name, changes in (
        ("stage9", {9: (3, 3, 3, 2, 2, 7, 2, 2, 4, 4, 2)}),
        ("stage12", {12: (3, 7, 3, 2, 2, 4, 4, 2)}),
        ("both", {9: (3, 3, 3, 2, 2, 7, 2, 2, 4, 4, 2),
                  12: (3, 7, 3, 2, 2, 4, 4, 2)}),
    ):
        groups = list(base7)
        for stage, keys in changes.items():
            groups[stage] = (keys, groups[stage][1])
        run(entry7, 7, name, groups)

    entry8, path8 = entry_and_path(env.clone(), campaign, 8)
    base8 = split(path8)
    for index, opening in enumerate((
        (1, 1, 1, 1, 4),
        (3, 1, 1, 1, 1),
        (3, 1, 1, 1, 4),
        (3, 3, 1, 1, 1),
        (3, 3, 1, 1, 4),
    )):
        groups = list(base8)
        groups[0] = (opening, groups[0][1])
        groups[10] = ((), ())
        run(entry8, 8, f"cheap_skip_{index}", groups)
    groups = list(base8)
    groups[0] = ((3, 3, 1, 1, 1), groups[0][1])
    groups[10] = ((1, 4), groups[10][1])
    run(entry8, 8, "cheap_realign", groups)


arena.run_program("lf52", probe)

"""Summarize key runs and coordinate-move effects in campaign levels 4--8."""

import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from perception import connected_components, safe_step


def normalize(action):
    return tuple(action) if isinstance(action, list) else action


def peg_count(frame):
    return sum(
        blob.color == 14 and blob.size == (4, 4)
        for blob in connected_components(frame, colors=(14,))
    )


def probe(env):
    with open("checkpoint.json") as stream:
        path = [normalize(action) for action in json.load(stream)["final_path"]]
    summaries = {}
    index = 0
    while index < len(path):
        level = int(env.levels_completed) + 1
        if level < 2:
            safe_step(env, path[index])
            index += 1
            continue
        keys = []
        groups = summaries.setdefault(level, [])
        while index < len(path) and isinstance(path[index], int):
            keys.append(path[index])
            safe_step(env, path[index])
            index += 1
            if int(env.levels_completed) >= level:
                break
        if keys:
            groups.append(("K", tuple(keys)))
        if int(env.levels_completed) >= level or index >= len(path):
            continue
        first = path[index]
        second = path[index + 1]
        before = peg_count(env.frame())
        safe_step(env, first)
        safe_step(env, second)
        after = peg_count(env.frame())
        groups.append(("C", first, second, before, after))
        index += 2
    for level in sorted(summaries):
        groups = summaries[level]
        print("level", level,
              "keys", sum(len(group[1]) for group in groups if group[0] == "K"),
              "moves", sum(group[0] == "C" for group in groups),
              "captures", sum(group[0] == "C" and group[4] < group[3]
                              for group in groups),
              "groups", tuple(groups), flush=True)


arena.run_program("lf52", probe)

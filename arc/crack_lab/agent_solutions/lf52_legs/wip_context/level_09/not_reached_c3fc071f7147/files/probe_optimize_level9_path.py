"""Apply full-reward local mutations to the integrated level-9 route."""

import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from legs import solve_repeated_frontier_bridge_carrier_peg_solitaire
from perception import safe_step


def normalize(action):
    return tuple(action) if isinstance(action, list) else action


class Recorder:
    def __init__(self, inner):
        self.inner = inner
        self.path = []

    @property
    def levels_completed(self):
        return self.inner.levels_completed

    def frame(self):
        return self.inner.frame()

    def clone(self):
        return self.inner.clone()

    def step(self, *args):
        self.path.append(args[0] if len(args) == 1 else tuple(args))
        return self.inner.step(*args)


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
    if keys:
        groups.append((tuple(keys), ()))
    return groups


def flatten(groups):
    return tuple(action for keys, clicks in groups for action in keys + clicks)


def probe(env):
    with open("checkpoint.json") as stream:
        campaign = tuple(normalize(action)
                         for action in json.load(stream)["final_path"])
    for action in campaign:
        safe_step(env, action)
    entry = env.clone()
    recorder = Recorder(entry.clone())
    solve_repeated_frontier_bridge_carrier_peg_solitaire(recorder)
    groups = split(tuple(recorder.path))
    print("reference", len(recorder.path), len(groups),
          tuple(len(keys) for keys, _ in groups), flush=True)

    def wins(candidate):
        node = entry.clone()
        for action in flatten(candidate):
            safe_step(node, action)
            if int(node.levels_completed) >= 9:
                return True
        return False

    mode = os.environ.get("OPT_MODE", "keys")
    selected = tuple(int(value) for value in os.environ.get(
        "OPT_GROUPS", ",".join(str(index) for index in range(len(groups)))
    ).split(","))
    successes = []
    tests = 0
    for group_index in selected:
        keys, clicks = groups[group_index]
        mutations = []
        if mode == "keys":
            mutations = [
                (index, keys[:index] + keys[index + 1:], clicks)
                for index in range(len(keys))
            ]
        elif mode == "macro":
            mutations = [("macro", keys, ())] if clicks else []
        elif mode == "half":
            mutations = [
                (index, keys, clicks[:index] + clicks[index + 1:])
                for index in range(len(clicks))
            ]
        elif mode == "destination" and clicks:
            destination = clicks[1]
            for dx, dy in ((-12, 0), (12, 0), (0, -12), (0, 12)):
                alternate = (6, destination[1] + dx,
                             destination[2] + dy)
                if 0 <= alternate[1] <= 63 and 0 <= alternate[2] <= 63:
                    mutations.append((alternate, keys, (clicks[0], alternate)))
        for label, replacement_keys, replacement_clicks in mutations:
            candidate = list(groups)
            candidate[group_index] = (replacement_keys, replacement_clicks)
            tests += 1
            if wins(candidate):
                successes.append((group_index, label, flatten(candidate)))
                print("success", tests, group_index, label,
                      len(flatten(candidate)), flush=True)
    print("done", tests, tuple((group, label, len(path))
                                for group, label, path in successes),
          flush=True)


arena.run_program("lf52", probe)

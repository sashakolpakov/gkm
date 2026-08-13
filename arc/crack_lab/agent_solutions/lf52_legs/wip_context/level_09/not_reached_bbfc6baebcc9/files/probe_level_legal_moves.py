"""List observed legal peg moves at each coordinate decision in a level path."""

import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from legs import (
    _bridge_carrier_moves,
    solve_compact_bridge_carrier_peg_solitaire,
)
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
    return tuple(groups)


class Recorder:
    def __init__(self, inner):
        self.inner = inner
        self.path = []

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def clone(self):
        return self.inner.clone()

    def step(self, action, *coordinates):
        recorded = ((action,) + coordinates) if coordinates else action
        self.path.append(recorded)
        return self.inner.step(action, *coordinates)


def probe(env):
    desired = int(os.environ.get("OPT_LEVEL", "5"))
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
    node = entry
    if os.environ.get("OPT_COMPACT") == "1":
        recorder = Recorder(entry.clone())
        solve_compact_bridge_carrier_peg_solitaire(recorder)
        level_path = tuple(recorder.path)
    else:
        level_path = campaign[start:end]
    for group_index, (keys, clicks) in enumerate(split(level_path)):
        for action in keys:
            safe_step(node, action)
        print(group_index, keys, clicks, _bridge_carrier_moves(node.frame()),
              flush=True)
        for action in clicks:
            safe_step(node, action)


arena.run_program("lf52", probe)

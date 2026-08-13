"""Replace one verified carrier-key run by a shorter run of action 7."""

import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from legs import solve_parallel_wrapped_bridge_carrier_peg_solitaire
from perception import safe_step


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


def flatten(groups):
    return tuple(action for keys, clicks in groups for action in keys + clicks)


def probe(env):
    desired = int(os.environ.get("OPT_LEVEL", "7"))
    with open("checkpoint.json") as stream:
        campaign = tuple(
            tuple(action) if isinstance(action, list) else action
            for action in json.load(stream)["final_path"]
        )
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

    if desired == 7:
        recorder = Recorder(entry.clone())
        solve_parallel_wrapped_bridge_carrier_peg_solitaire(recorder)
        path = tuple(recorder.path)
    else:
        path = campaign[start:end]
    groups = split(path)
    first = int(os.environ.get("OPT_FIRST_GROUP", "0"))
    last = min(len(groups), int(os.environ.get(
        "OPT_LAST_GROUP", str(len(groups))
    )))
    wins = []
    tests = 0
    for stage in range(first, last):
        keys, clicks = groups[stage]
        for count in range(1, len(keys)):
            candidate = list(groups)
            candidate[stage] = ((7,) * count, clicks)
            node = entry.clone()
            path = flatten(candidate)
            for action in path:
                safe_step(node, action)
                if int(node.levels_completed) >= desired:
                    wins.append((len(path), stage, count, path))
                    break
            tests += 1
        print("reset_run_progress", stage, tests, len(wins), flush=True)
    print("reset_run_result", desired, tests,
          tuple((cost, stage, count) for cost, stage, count, _ in wins),
          flush=True)
    if wins:
        print("reset_run_best", min(wins), flush=True)


arena.run_program("lf52", probe)

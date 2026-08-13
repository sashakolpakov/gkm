"""Replace part of one verified controller run with fewer action-7 steps."""

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
    desired = 7
    with open("checkpoint.json") as stream:
        campaign = tuple(
            tuple(action) if isinstance(action, list) else action
            for action in json.load(stream)["final_path"]
        )
    prior = int(env.levels_completed)
    entry = None
    for action in campaign:
        safe_step(env, action)
        current = int(env.levels_completed)
        if prior < desired - 1 <= current:
            entry = env.clone()
            break
        prior = current

    recorder = Recorder(entry.clone())
    solve_parallel_wrapped_bridge_carrier_peg_solitaire(recorder)
    path = tuple(recorder.path)
    groups = split(path)
    stage = int(os.environ.get("OPT_GROUP", "22"))
    original, clicks = groups[stage]

    candidates = []
    seen = set()
    for left in range(len(original)):
        for right in range(left + 2, len(original) + 1):
            for count in range(1, right - left):
                keys = original[:left] + (7,) * count + original[right:]
                if keys in seen:
                    continue
                seen.add(keys)
                candidates.append((len(original) - len(keys), left,
                                   right, count, keys))
    candidates.sort(key=lambda item: (-item[0], item[1], item[2], item[3]))
    batch_start = int(os.environ.get("OPT_BATCH_START", "0"))
    batch_end = min(len(candidates), int(os.environ.get(
        "OPT_BATCH_END", str(len(candidates))
    )))

    prefix = entry.clone()
    for action in flatten(groups[:stage]):
        safe_step(prefix, action)
    suffix = flatten(groups[stage + 1:])
    wins = []
    for index in range(batch_start, batch_end):
        saving, left, right, count, keys = candidates[index]
        child = prefix.clone()
        won = False
        for action in keys + clicks + suffix:
            safe_step(child, action)
            if int(child.levels_completed) >= desired:
                won = True
                break
        if won:
            candidate_groups = list(groups)
            candidate_groups[stage] = (keys, clicks)
            full_path = flatten(candidate_groups)
            wins.append((len(full_path), index, saving, left, right,
                         count, keys, full_path))
            print("mixed_reset_win", stage, index, len(full_path),
                  saving, left, right, count, keys, flush=True)
    print("mixed_reset_result", stage, len(original), original,
          len(candidates), (batch_start, batch_end), len(wins), flush=True)
    if wins:
        print("mixed_reset_best", min(wins), flush=True)


arena.run_program("lf52", probe)

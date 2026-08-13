"""Count public-frame no-op actions in the reproduced campaign prefix."""

import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from perception import arr, safe_step


def normalize(action):
    return tuple(action) if isinstance(action, list) else action


def probe(env):
    with open("checkpoint.json") as stream:
        path = [normalize(action) for action in json.load(stream)["final_path"]]
    counts = {}
    noops = {}
    boundaries = []
    for index, action in enumerate(path):
        level = int(env.levels_completed) + 1
        before = arr(env.frame())[1:, :].copy()
        old_completed = int(env.levels_completed)
        safe_step(env, action)
        changed = int((arr(env.frame())[1:, :] != before).sum())
        counts[level] = counts.get(level, 0) + 1
        if changed == 0:
            noops.setdefault(level, []).append((index, action))
        if env.levels_completed > old_completed:
            boundaries.append((level, index + 1))
    print("counts", counts, "boundaries", boundaries, flush=True)
    print("noops", {level: tuple(items) for level, items in noops.items()},
          flush=True)


arena.run_program("lf52", probe)

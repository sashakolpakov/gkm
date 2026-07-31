import json
import sys
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from legs import clone_after, fast_reach


def visible_key(node):
    frame = np.asarray(node.frame())[:62]
    return np.where(np.isin(frame, (1, 8, 9, 11, 14, 15)), frame, 0).tobytes()


def search(env, max_stages=12, max_expand=10000, histories_per_key=12):
    base = int(env.levels_completed)
    frontier = [(env.clone(), [])]
    expanded = 0
    for stage in range(max_stages + 1):
        groups = defaultdict(list)
        print("stage", stage, "frontier", len(frontier), "expanded", expanded)
        for node, prefix in frontier:
            expanded += 1
            reward_path, reach = fast_reach(node)
            if reward_path is not None:
                return prefix + reward_path
            for pos, walk in sorted(reach.items(), key=lambda item: len(item[1])):
                macro = walk + [5]
                child = clone_after(node, macro)
                combined = prefix + macro
                if int(child.levels_completed) > base:
                    return combined
                groups[visible_key(child)].append((child, combined))
            if expanded >= max_expand:
                return None
        frontier = []
        for candidates in groups.values():
            candidates.sort(key=lambda item: len(item[1]))
            frontier.extend(candidates[:histories_per_key])
    return None


def probe(env):
    with open("checkpoint.json") as fh:
        path = json.load(fh)["final_path"]
    for action in path:
        env.step(action)
    started = time.time()
    plan = search(env)
    print("macro_search", round(time.time() - started, 3), plan)


levels, path, err = arena.run_program("g50t", probe)
print("probe_result", levels, len(path), err)

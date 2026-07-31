import sys
from collections import deque

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import numpy as np
import gkm_arena as arena

import players
from probe_clean_connect import (
    component_count,
    occupied_mask,
    selection_roots,
    shortest_clean_merge,
)


def clean_merges(root, occupied, components, max_states=20000, max_depth=32):
    queue = deque([(root.clone(), [])])
    seen = {np.asarray(root.frame()).tobytes()}
    goals = []
    while queue and len(seen) <= max_states:
        node, path = queue.popleft()
        if node.levels_completed >= 6 or component_count(node.frame()) < components:
            goals.append((path, node))
            continue
        if len(path) >= max_depth:
            continue
        for action in range(1, 6):
            child = node.clone()
            child.step(action)
            frame = child.frame()
            if int(occupied_mask(frame).sum()) != occupied:
                continue
            key = np.asarray(frame).tobytes()
            if key in seen:
                continue
            seen.add(key)
            queue.append((child, path + [action]))
    return goals


def probe(env):
    while env.levels_completed < 5:
        getattr(players, f"play_level_{env.levels_completed + 1}")(env)
    occupied = int(occupied_mask(env.frame()).sum())
    candidates = []
    for click, root in selection_roots(env):
        goals = clean_merges(root, occupied, component_count(root.frame()))
        print("FIRST", click, "goals", [path for path, _ in goals], flush=True)
        for path, node in goals:
            candidates.append(([] if click is None else [(6, *click)], path, node))
    print("CANDIDATES", len(candidates), flush=True)
    for index, (prefix, path, node) in enumerate(candidates):
        for click, root in selection_roots(node):
            next_path, result = shortest_clean_merge(
                root, occupied, component_count(node.frame()),
                max_states=4000, max_depth=32,
            )
            if next_path is not None:
                second_prefix = [] if click is None else [(6, *click)]
                print("VIABLE", index, prefix + path + second_prefix + next_path,
                      "components", component_count(result.frame()), flush=True)
                return
        if index % 10 == 0:
            print("CHECKED", index, flush=True)
    print("NO_VIABLE")


arena.run_program("cn04", probe)

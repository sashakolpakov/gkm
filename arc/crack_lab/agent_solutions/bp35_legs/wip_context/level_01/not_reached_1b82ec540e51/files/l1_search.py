import sys
from collections import deque

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A

from perception import connected_components


MAX_STATES = 30000
MAX_DEPTH = 36


def observation_key(env):
    # The last row only records action history and is not physical state.
    return np.asarray(env.frame())[:63].tobytes()


def candidates(env):
    actions = [3, 4]
    for blob in connected_components(env.frame(), colors=(14,), min_area=4):
        x = round(blob.centroid[1])
        y = round(blob.centroid[0])
        actions.append((6, x, y))
    return actions


def apply(env, action):
    child = env.clone()
    if isinstance(action, tuple):
        child.step(*action)
    else:
        child.step(action)
    return child


def metrics(env):
    frame = np.asarray(env.frame())
    yellow = int(np.count_nonzero(frame == 14))
    traversable = int(np.count_nonzero(frame == 10))
    body = np.argwhere(frame == 9)
    body_col = int(body[:, 1].mean()) if len(body) else -1
    return yellow, traversable, body_col


def search(env):
    queue = deque([(env.clone(), [])])
    seen = {observation_key(env)}
    best = metrics(env)
    best_path = []
    expanded = 0
    while queue and len(seen) < MAX_STATES:
        node, path = queue.popleft()
        expanded += 1
        if node.levels_completed:
            print("FOUND", path)
            print("SEARCH", "seen", len(seen), "expanded", expanded, "depth", len(path))
            return path
        if len(path) >= MAX_DEPTH or node.terminal():
            continue
        for action in candidates(node):
            child = apply(node, action)
            key = observation_key(child)
            if key in seen:
                continue
            seen.add(key)
            child_path = path + [action]
            score = metrics(child)
            if (score[1], -score[0], score[2]) > (best[1], -best[0], best[2]):
                best = score
                best_path = child_path
            if child.levels_completed:
                print("FOUND", child_path)
                print("SEARCH", "seen", len(seen), "expanded", expanded, "depth", len(child_path))
                return child_path
            queue.append((child, child_path))
        if expanded % 2000 == 0:
            print(
                "PROGRESS",
                "seen",
                len(seen),
                "expanded",
                expanded,
                "queue",
                len(queue),
                "depth",
                len(path),
                "best",
                best,
                "best_path",
                best_path,
            )
    print(
        "EXHAUSTED",
        "seen",
        len(seen),
        "expanded",
        expanded,
        "queue",
        len(queue),
        "best",
        best,
        "best_path",
        best_path,
    )
    return None


def probe(env):
    search(env)


print("run", A.run_program("bp35", probe))

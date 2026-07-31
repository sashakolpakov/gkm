"""Replay-based bounded search for bp35 level 4."""
from collections import deque
import json
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

from legs import COL_ANCHORS, ROW_ANCHORS


PREFIX = json.load(open("level4_prefix.json"))
INTERACTIVE = (8, 14, 15)


def advance(env):
    for action in PREFIX:
        env.step(*(action if isinstance(action, list) else [action]))
    assert env.levels_completed == 3


def key(env):
    frame = np.asarray(env.frame())
    counter = int(np.count_nonzero(frame[63]))
    return frame[:63].tobytes(), counter % 2


def choices(env):
    frame = env.frame()
    out = [(3,), (4,)]
    for y in ROW_ANCHORS:
        for x in COL_ANCHORS:
            if int(frame[y][x]) in INTERACTIVE:
                out.append((6, x, y))
    return out


def search(env, max_states=6000, max_depth=55):
    root = env.clone()

    def reconstruct(path):
        node = root.clone()
        for action in path:
            node.step(*action)
        return node

    queue = deque([()])
    seen = {key(root)}
    expansions = 0
    deepest = 0
    while queue and len(seen) < max_states:
        path = queue.popleft()
        node = reconstruct(path)
        expansions += 1
        if len(path) > deepest:
            deepest = len(path)
            if deepest % 5 == 0:
                print("DEPTH", deepest, "exp", expansions, "seen", len(seen))
        if len(path) >= max_depth or node.terminal():
            continue
        for action in choices(node):
            child = node.clone()
            child.step(*action)
            new_path = path + (action,)
            if child.levels_completed > 3:
                print("FOUND", len(new_path), "exp", expansions, "seen", len(seen))
                print("PATH", list(new_path))
                return list(new_path)
            if child.terminal():
                continue
            state = key(child)
            if state in seen:
                continue
            seen.add(state)
            queue.append(new_path)
    print("MISS", expansions, len(seen), "deepest", deepest)
    return []


def probe(env):
    advance(env)
    search(env)


print(A.run_program("bp35", probe))

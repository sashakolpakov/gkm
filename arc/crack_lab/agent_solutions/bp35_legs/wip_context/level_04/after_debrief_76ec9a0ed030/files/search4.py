"""Bounded symbolic search over level-4's observed move/click affordances."""
from collections import deque
import json
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

from perception import connected_components
from legs import COL_ANCHORS, ROW_ANCHORS


PREFIX = json.load(open("level4_prefix.json"))


def advance(env):
    for action in PREFIX:
        env.step(*(action if isinstance(action, list) else [action]))
    assert env.levels_completed == 3


def key(env):
    frame = np.asarray(env.frame())
    return (frame[:63].tobytes(), sum(int(v) != 0 for v in frame[63]) % 2)


def actions(env):
    out = [(3,), (4,)]
    frame = env.frame()
    for y in ROW_ANCHORS:
        for x in COL_ANCHORS:
            if int(frame[y][x]) not in (0, 3, 5, 9, 10, 11):
                out.append((6, x, y))
    return out


def search(env, max_states=12000, max_depth=70):
    root = env.clone()
    q = deque([(root, [])])
    seen = {key(root)}
    best = (0, [])
    expansions = 0
    while q and len(seen) < max_states:
        node, path = q.popleft()
        expansions += 1
        if len(path) >= max_depth:
            continue
        for action in actions(node):
            child = node.clone()
            child.step(*action)
            new_path = path + [action]
            if child.levels_completed > 3:
                print("FOUND", len(new_path), "exp", expansions, "seen", len(seen))
                print("PATH", new_path)
                return new_path
            if child.terminal():
                continue
            k = key(child)
            if k in seen:
                continue
            seen.add(k)
            q.append((child, new_path))
            # Dense progress: reward discovery of new camera/world layouts.
            score = len(set(np.asarray(child.frame())[:63].ravel().tolist()))
            if score > best[0]:
                best = (score, new_path)
        if expansions % 1000 == 0:
            print("PROGRESS", expansions, "seen", len(seen), "depth", len(path),
                  "queue", len(q), "best", best[0])
    print("MISS", expansions, len(seen), "best_path", best[1])
    return None


def probe(env):
    advance(env)
    search(env)


print(A.run_program("bp35", probe))

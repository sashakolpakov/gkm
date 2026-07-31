"""Context-bounded replay search for bp35 level 4."""
from collections import deque
import json
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

from legs import COL_ANCHORS, ROW_ANCHORS
from perception import connected_components


PREFIX = json.load(open("level4_prefix.json"))


def advance(env):
    for action in PREFIX:
        env.step(*(action if isinstance(action, list) else [action]))
    assert env.levels_completed == 3


def avatar_cell(frame):
    avatars = connected_components(frame, colors=(9,), min_area=4)
    if not avatars:
        return None
    r, c = avatars[0].centroid
    i = min(range(len(ROW_ANCHORS)), key=lambda k: abs(ROW_ANCHORS[k] - r))
    j = min(range(len(COL_ANCHORS)), key=lambda k: abs(COL_ANCHORS[k] - c))
    return i, j


def choices(env):
    frame = env.frame()
    avatar = avatar_cell(frame)
    out = [(3,), (4,)]
    if avatar is None:
        return out
    ai, aj = avatar
    for i, y in enumerate(ROW_ANCHORS):
        for j, x in enumerate(COL_ANCHORS):
            color = int(frame[y][x])
            if color == 8:
                out.append((6, x, y))
            elif color == 14 and j == aj and abs(i - ai) <= 1:
                out.append((6, x, y))
            elif color == 15 and abs(i - ai) <= 1 and abs(j - aj) <= 1:
                out.append((6, x, y))
    return out


def state_key(env):
    frame = np.asarray(env.frame())
    return frame[:63].tobytes(), int(np.count_nonzero(frame[63])) % 2


def search(env, max_states=900, max_depth=48):
    root = env.clone()

    def reconstruct(path):
        node = root.clone()
        for action in path:
            node.step(*action)
        return node

    queue = deque([()])
    seen = {state_key(root)}
    expansions = 0
    deepest = 0
    while queue and len(seen) < max_states:
        path = queue.popleft()
        node = reconstruct(path)
        expansions += 1
        if len(path) > deepest:
            deepest = len(path)
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
            key = state_key(child)
            if key in seen:
                continue
            seen.add(key)
            queue.append(new_path)
    print("MISS", expansions, len(seen), "deepest", deepest)
    return []


def probe(env):
    advance(env)
    search(env)


print(A.run_program("bp35", probe))

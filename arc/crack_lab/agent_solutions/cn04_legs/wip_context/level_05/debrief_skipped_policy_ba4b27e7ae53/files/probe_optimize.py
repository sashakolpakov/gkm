from collections import deque

import numpy as np

import gkm_try as harness
from perception import frame_delta


def progress(env):
    frame = np.asarray(env.frame())
    return 315 - int(np.count_nonzero(frame[1:] != 15))


def improve(start, max_states=3000, max_depth=24):
    q = deque([(start.clone(), [])])
    seen = {np.asarray(start.frame()).tobytes()}
    best = (progress(start), [], start.clone())
    while q and len(seen) < max_states:
        node, path = q.popleft()
        if node.levels_completed > 4:
            print("WIN", path)
            return node, path
        score = progress(node)
        if score > best[0]:
            best = (score, path, node.clone())
            print("IMPROVE", score, path)
        if node.levels_completed > 4 or len(path) >= max_depth:
            continue
        for action in (1, 2, 3, 4, 5):
            child = node.clone()
            child.step(action)
            key = np.asarray(child.frame()).tobytes()
            if key not in seen:
                seen.add(key)
                q.append((child, path + [action]))
    print("SEARCHED", len(seen), "BEST", best[0], best[1])
    return best[2], best[1]


def probe(env):
    harness.resumed_solve(env)
    p1 = [2] * 8 + [3] * 7
    node = env.clone()
    for action in p1:
        node.step(action)
    node.step(6, 47, 47)
    p2 = [1] * 12
    for action in p2:
        node.step(action)
    node.step(6, 12, 39)
    node, p3 = improve(node, 3000, 24)
    print("FINAL", node.levels_completed, progress(node), p1, p2, p3)


harness.A.run_program("cn04", probe)

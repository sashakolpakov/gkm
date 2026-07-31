import sys
from collections import deque

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A

import perception as P
import players


def key(env):
    frame = P.arr(env.frame()).copy()
    frame[np.isin(frame, (0, 6, 7))] = 6
    return frame.tobytes()


def actions(env):
    result = [1, 2, 3, 4, 5]
    for obj in P.connected_components(env.frame(), colors=(1, 9), min_area=2):
        r0, c0, r1, c1 = obj.bbox
        result.append((6, (c0 + c1) // 2, (r0 + r1) // 2))
    return result


def run(env):
    for level in range(1, 6):
        getattr(players, f"play_level_{level}")(env)
    queue = deque([(env.clone(), [])])
    seen = {key(env)}
    limit = 30000
    while queue and len(seen) <= limit:
        node, path = queue.popleft()
        if len(path) >= 45 or node.terminal():
            continue
        for action in actions(node):
            child = node.clone()
            if isinstance(action, tuple):
                child.step(*action)
            else:
                child.step(action)
            child_path = path + [action]
            if child.levels_completed > 5:
                print("FOUND", child_path)
                return
            child_key = key(child)
            if child_key in seen:
                continue
            seen.add(child_key)
            queue.append((child, child_path))
    print("NOT_FOUND", len(seen))


A.run_program("m0r0", run)

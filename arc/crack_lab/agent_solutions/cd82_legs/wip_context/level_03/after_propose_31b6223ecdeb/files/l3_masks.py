from collections import deque

import numpy as np

import gkm_try as harness


DIRS = (1, 2, 3, 4)
NAMES = {1: "U", 2: "D", 3: "L", 4: "R"}


def text(mask):
    return "/".join("".join("#" if value else "." for value in row) for row in mask)


def observe(env):
    harness.m.solve(env)
    queue = deque([(env.clone(), [])])
    seen = set()
    found = []
    while queue and len(seen) < 40:
        node, path = queue.popleft()
        key = np.asarray(node.frame())[:-1].tobytes()
        if key in seen:
            continue
        seen.add(key)
        applied = node.clone()
        applied.step(5)
        canvas = np.asarray(applied.frame())[34:44, 27:37]
        found.append((path, canvas != 0))
        if len(path) < 10:
            for action in DIRS:
                child = node.clone()
                child.step(action)
                queue.append((child, path + [action]))
    print("positions", len(found))
    for path, mask in found:
        print("".join(NAMES[a] for a in path) or "START", int(mask.sum()), text(mask))


harness.A.run_program("cd82", observe)

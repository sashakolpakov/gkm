"""Bounded reward BFS over level 7's two observed coordinate controls."""
from collections import deque
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

from solve import solve


CONTROLS = ((22, 34), (32, 42))
POSITIONS = (
    (23, 20), (23, 23), (23, 26), (23, 29),
    (23, 32), (23, 35), (23, 38), (23, 41),
    (26, 20), (29, 20),
    (35, 29), (35, 32), (38, 32), (38, 29),
)


def key(env):
    frame = np.asarray(env.frame())
    return tuple(int(frame[p]) for p in POSITIONS)


def run(env):
    solve(env)
    base_level = env.levels_completed
    queue = deque([(env.clone(), ())])
    seen = {key(env)}
    depth = -1
    while queue and len(seen) < 150000:
        node, path = queue.popleft()
        if len(path) != depth:
            depth = len(path)
            print("depth", depth, "seen", len(seen), "queue", len(queue),
                  flush=True)
        for op, control in enumerate(CONTROLS):
            child = node.clone()
            child.step(6, *control)
            child_path = path + (op,)
            if child.levels_completed > base_level:
                print("FOUND", child_path,
                      tuple(CONTROLS[i] for i in child_path), flush=True)
                return
            child_key = key(child)
            if child_key not in seen and not child.terminal():
                seen.add(child_key)
                queue.append((child, child_path))
    print("stopped", len(seen), "queue", len(queue), flush=True)


if __name__ == "__main__":
    A.run_program("lp85", run)

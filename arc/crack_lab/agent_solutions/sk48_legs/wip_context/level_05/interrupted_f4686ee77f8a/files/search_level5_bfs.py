"""Bounded direct-clone BFS for the level-5 reward transition."""
from collections import deque
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

import perception as p
import players


def key(env):
    data = p.arr(env.frame())
    return data.tobytes()


def run(env):
    for level in range(1, 5):
        getattr(players, f"play_level_{level}")(env)

    root_level = env.levels_completed
    queue = deque([(env.clone(), ())])
    seen = {key(env)}
    expanded = 0
    max_states = 20000
    max_depth = 90
    while queue and len(seen) < max_states:
        node, path = queue.popleft()
        expanded += 1
        if len(path) >= max_depth:
            continue
        for action in (1, 2, 3, 4):
            child = node.clone()
            child.step(action)
            child_path = path + (action,)
            if child.levels_completed > root_level:
                print("FOUND", list(child_path), expanded, len(seen))
                return
            child_key = key(child)
            if child_key in seen:
                continue
            seen.add(child_key)
            queue.append((child, child_path))
        if expanded % 2000 == 0:
            print("PROGRESS", expanded, len(seen), len(path))
    print("EXHAUSTED", expanded, len(seen))


levels, path, error = arena.run_program("sk48", run)
print("END", levels, len(path), error)

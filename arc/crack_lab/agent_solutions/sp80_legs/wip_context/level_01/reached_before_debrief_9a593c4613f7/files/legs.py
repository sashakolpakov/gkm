# Shared leg library: small, named, reusable skills.
# Players import from here; add a NEW leg only when no existing leg fits.

import numpy as np
from collections import deque


def play_fixed_sequence(env, sequence):
    """Play a fixed list of actions on the real env."""
    for act in sequence:
        if env.terminal():
            break
        env.step(int(act))


def bfs_win(env, actions=(1, 2, 3, 4, 5, 6), max_states=5000, max_depth=40):
    """BFS over clone states to find shortest action sequence that increases
    env.levels_completed.  Returns the path (list of ints) or None."""
    base_level = env.levels_completed

    def key_fn(e):
        f = np.array(e.frame())
        return f.tobytes()

    start_key = key_fn(env)
    q = deque([(env.clone(), [])])
    seen = {start_key}

    while q and len(seen) <= max_states:
        node, path = q.popleft()
        if len(path) >= max_depth:
            continue
        for action in actions:
            child = node.clone()
            child.step(int(action))
            if child.levels_completed > base_level:
                return path + [int(action)]
            k = key_fn(child)
            if k in seen:
                continue
            seen.add(k)
            q.append((child, path + [int(action)]))
    return None


def bfs_win_compact(env, key_fn, actions=(1, 2, 3, 4, 5, 6), max_states=5000, max_depth=40):
    """BFS with a custom key function (cheaper than full frame hash)."""
    base_level = env.levels_completed
    start_key = key_fn(env)
    q = deque([(env.clone(), [])])
    seen = {start_key}

    while q and len(seen) <= max_states:
        node, path = q.popleft()
        if len(path) >= max_depth:
            continue
        for action in actions:
            child = node.clone()
            child.step(int(action))
            if child.levels_completed > base_level:
                return path + [int(action)]
            k = key_fn(child)
            if k is None or k in seen:
                continue
            seen.add(k)
            q.append((child, path + [int(action)]))
    return None

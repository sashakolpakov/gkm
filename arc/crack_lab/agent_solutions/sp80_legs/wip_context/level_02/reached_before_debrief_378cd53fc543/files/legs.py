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


def click_select(env, x, y):
    """Coordinate interaction: issue action 6 at pixel (x=col, y=row).
    In sp80 this selects/grabs the object under the cursor so that the
    directional actions (1..4) then move THAT object.  Recorded in replay
    paths as [6, x, y]."""
    env.step(6, int(x), int(y))


def make_bbox_key(color):
    """Return a compact key function that tracks the bounding box of pixels
    with the given color value.  Returns None if the color is absent (BFS
    treats that as an already-seen / skip state)."""
    def key_fn(env):
        f = np.array(env.frame())
        rows, cols = np.where(f == color)
        if len(rows) == 0:
            return None
        return (int(rows.min()), int(rows.max()), int(cols.min()), int(cols.max()))
    return key_fn


def bfs_or_fallback(env, key_fn, fallback,
                    actions=(1, 2, 3, 4, 5, 6),
                    max_states=500, max_depth=20):
    """Try BFS (compact key) to find a winning path; play it.
    If BFS finds nothing, play *fallback* (a fixed action list) instead."""
    path = bfs_win_compact(env, key_fn, actions=actions,
                           max_states=max_states, max_depth=max_depth)
    play_fixed_sequence(env, path if path else fallback)


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

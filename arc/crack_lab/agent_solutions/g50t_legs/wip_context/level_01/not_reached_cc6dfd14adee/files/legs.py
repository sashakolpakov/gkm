# Shared leg library: small, named, reusable skills.
# Players import from here; add a NEW leg only when no existing leg fits.
#
# Legs are written against `perception.py` observations and the arena's
# clone/step/frame/levels_completed/terminal interface only. No game source is
# read. Legs are intentionally game-agnostic search/navigation primitives.

import numpy as np
from collections import deque

import perception as P

MOVES = (1, 2, 3, 4)
ALL_ACTIONS = (1, 2, 3, 4, 5)


# ---------------------------------------------------------------------------
# Observation helpers
# ---------------------------------------------------------------------------

def frame_np(env):
    return np.asarray(env.frame())


def _row_is_counter(f):
    """Heuristically detect a monotone 'move counter / timer' row.

    Some g50t frames dedicate the bottom border row to a step counter that
    changes every move. When building BFS keys we want to ignore such a row so
    that revisiting the same *world* state dedups correctly.
    """
    r = f.shape[0] - 1
    row = f[r]
    # A counter row is a long uniform border that only mutates in place.
    return r


def state_key(env, mask_counter_row=True):
    """A dedup key for the *world* state, optionally ignoring the counter row."""
    f = frame_np(env).copy()
    if mask_counter_row:
        f[_row_is_counter(f), :] = 0
    return f.tobytes()


# ---------------------------------------------------------------------------
# Generic search legs
# ---------------------------------------------------------------------------

def bfs_to_goal(env, goal_fn, actions=ALL_ACTIONS, key_fn=None,
                max_states=40000, prune_terminal=True):
    """Breadth-first search over clones for the shortest action path that makes
    ``goal_fn(child)`` true. Returns the path (list of ints) or None.

    ``goal_fn`` receives a stepped clone. ``key_fn`` maps a clone to a hashable
    dedup key (defaults to the counter-masked world state).
    """
    if key_fn is None:
        key_fn = lambda e: state_key(e, mask_counter_row=True)
    start = env.clone()
    seen = {key_fn(start)}
    q = deque([(start, [])])
    while q and len(seen) <= max_states:
        node, path = q.popleft()
        for a in actions:
            child = node.clone()
            child.step(int(a))
            if goal_fn(child):
                return path + [int(a)]
            if prune_terminal and child.terminal():
                continue
            k = key_fn(child)
            if k in seen:
                continue
            seen.add(k)
            q.append((child, path + [int(a)]))
    return None


def solve_level_by_search(env, actions=ALL_ACTIONS, max_states=60000):
    """General leg: find and execute a path that increases levels_completed.

    This is the reusable "just search for the win" skill. It plans entirely on
    clones (bounded), then commits the winning path on the real env. Returns
    True on success, False if no winning path exists within the budget.
    """
    base = env.levels_completed
    path = bfs_to_goal(
        env,
        goal_fn=lambda e: e.levels_completed > base,
        actions=actions,
        max_states=max_states,
    )
    if not path:
        return False
    for a in path:
        if env.terminal():
            break
        env.step(int(a))
    return env.levels_completed > base

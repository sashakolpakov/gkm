# Shared leg library: small, named, reusable skills.
# Players import from here; add a NEW leg only when no existing leg fits.
from collections import deque


def find_winning_path(env, actions=(1, 2, 3, 4), max_states=8000):
    """BFS over env.clone()s from the current state; returns the shortest
    action sequence that increases env.levels_completed, or None if no such
    sequence is found within max_states distinct frames explored. Dedupes
    states by raw frame bytes, so it works regardless of what the frame
    contains -- no game-specific assumptions. Does not touch the real env."""
    start = env.clone()
    key0 = start.frame().tobytes()
    seen = {key0}
    q = deque([(start, [])])
    base_level = env.levels_completed
    while q:
        node, path = q.popleft()
        for a in actions:
            c = node.clone()
            c.step(a)
            if c.levels_completed > base_level:
                return path + [a]
            key = c.frame().tobytes()
            if key not in seen:
                seen.add(key)
                q.append((c, path + [a]))
        if len(seen) >= max_states:
            return None
    return None


def execute_path(env, path):
    """Commit a precomputed action sequence to the real env."""
    for a in path:
        if env.terminal():
            break
        env.step(a)


def solve_by_search(env, actions=(1, 2, 3, 4), max_states=8000):
    """Compose find_winning_path + execute_path: search on clones for a
    sequence that raises levels_completed, then replay it on the real env.
    Good default for any level whose winning state is reachable within a
    small (thousands of frames) BFS horizon."""
    path = find_winning_path(env, actions=actions, max_states=max_states)
    if path:
        execute_path(env, path)
    return path

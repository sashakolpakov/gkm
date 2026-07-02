import numpy as np
from collections import deque


def _fh(f):
    # Hash the play-field + side state displays, but EXCLUDE the bottom timer bar
    # (rows 61-62 advance every move and would make every state look unique).
    return f[0:61, :].tobytes()


def _find_win(env, max_nodes=60000):
    """BFS over clones to find an action sequence that increases levels_completed.
    State identity = field hash (so blocked moves and revisits are pruned), keeping
    the search bounded to the reachable distinct configurations."""
    base = env.levels_completed
    start = env.clone()
    seen = {_fh(start.frame())}
    q = deque([(start, [])])
    n = 0
    while q and n < max_nodes:
        node, path = q.popleft()
        for a in (1, 2, 3, 4):
            c = node.clone()
            c.step(a)
            n += 1
            if c.levels_completed > base:
                return path + [a]
            h = _fh(c.frame())
            if h not in seen:
                seen.add(h)
                q.append((c, path + [a]))
            if n >= max_nodes:
                break
    return None


def solve(env):
    # Clear levels one at a time: search on clones for a sequence that bumps the
    # reward, commit it on the real env, then repeat for the next level.
    moves = 0
    move_cap = 580
    stuck = 0
    while not env.terminal() and moves < move_cap and stuck < 3:
        target = env.levels_completed
        seq = _find_win(env)
        if not seq:
            stuck += 1
            continue
        for a in seq:
            if env.terminal() or moves >= move_cap:
                break
            env.step(a)
            moves += 1
        if env.levels_completed <= target:
            stuck += 1
        else:
            stuck = 0

# Shared leg library: small, named, reusable skills.
# Players import from here; add a NEW leg only when no existing leg fits.
import numpy as np

ACTIONS = (1, 2, 3, 4)


def _find_noise_mask(env, grow_steps=120):
    """Frames often carry a HUD/counter overlay that changes every step
    regardless of what the avatar does (e.g. a "moves used" bar). Detect it
    generically: probe the action with the SMALLEST immediate pixel diff
    (most likely a blocked/no-op move for the avatar) and repeat it on a
    clone to reveal every cell that drifts on its own, independent of
    movement. Returns a boolean mask of those cells."""
    base = env.clone()
    f0 = base.frame()
    diffs = {}
    for a in ACTIONS:
        c = base.clone()
        try:
            f = c.step(a)
        except Exception:
            continue
        diffs[a] = int((f0 != f).sum())
    if not diffs:
        return np.zeros(f0.shape, dtype=bool)
    noop_a = min(diffs, key=diffs.get)
    mask = np.zeros(f0.shape, dtype=bool)
    c = base.clone()
    for _ in range(grow_steps):
        if c.terminal():
            break
        f = c.step(noop_a)
        mask |= (f != f0)
    return mask


def _masked_bytes(f, mask):
    g = f.copy()
    g[mask] = -1
    return g.tobytes()


def bfs_to_level_up(env, max_nodes=6000):
    """Generic blind search leg: BFS over clones of env, deduplicating
    states by frame content with the HUD/counter noise masked out, until
    levels_completed increases. Commits the shortest winning action
    sequence found to the REAL env (via env.step) and returns True on
    success, False if no solution was found within max_nodes expansions."""
    base_level = env.levels_completed
    root = env.clone()
    mask = _find_noise_mask(root)
    visited = {_masked_bytes(root.frame(), mask)}
    frontier = [(root, [])]
    nodes = 0
    head = 0
    while head < len(frontier) and nodes < max_nodes:
        node_env, path = frontier[head]
        head += 1
        for a in ACTIONS:
            c = node_env.clone()
            try:
                f = c.step(a)
            except Exception:
                continue
            nodes += 1
            if c.levels_completed > base_level:
                for act in path + [a]:
                    env.step(act)
                return True
            if c.terminal():
                continue
            key = _masked_bytes(f, mask)
            if key not in visited:
                visited.add(key)
                frontier.append((c, path + [a]))
    return False

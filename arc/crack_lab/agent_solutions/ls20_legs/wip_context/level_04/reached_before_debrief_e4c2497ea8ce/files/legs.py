# Shared leg library: small, named, reusable skills.
# Players import from here; add a NEW leg only when no existing leg fits.
from collections import deque


def detect_noise_mask(env, actions=(1, 2, 3, 4), grow_steps=120):
    """Some levels overlay a HUD/counter that drifts every step regardless of
    what the avatar does (e.g. a "moves used" bar); raw-frame dedup then
    treats every frame as novel and a BFS never revisits a state. Detect the
    noise generically and game-agnostically: probe each action once, take the
    one causing the SMALLEST pixel change (likely a blocked/no-op move), then
    repeat just that action on a clone for grow_steps ticks -- any cell that
    changes despite the "do nothing useful" action is noise, not avatar
    state. Returns a boolean mask (frame-shaped) of those cells; all-False if
    no noise is found. Does not touch the real env."""
    import numpy as np
    base = env.clone()
    f0 = base.frame()
    diffs = {}
    for a in actions:
        c = base.clone()
        try:
            f = c.step(a)
        except Exception:
            continue
        diffs[a] = int((f0 != f).sum())
    mask = np.zeros(f0.shape, dtype=bool)
    if not diffs:
        return mask
    noop_a = min(diffs, key=diffs.get)
    c = base.clone()
    for _ in range(grow_steps):
        if c.terminal():
            break
        f = c.step(noop_a)
        mask |= (f != f0)
    return mask


def _masked_bytes(frame, mask):
    if mask is None or not mask.any():
        return frame.tobytes()
    g = frame.copy()
    g[mask] = -1
    return g.tobytes()


def find_winning_path(env, actions=(1, 2, 3, 4), max_states=8000, mask=None):
    """BFS over env.clone()s from the current state; returns the shortest
    action sequence that increases env.levels_completed, or None if no such
    sequence is found within max_states distinct frames explored. Dedupes
    states by raw frame bytes (optionally with `mask` cells -- see
    detect_noise_mask -- ignored, for levels with a HUD that drifts on its
    own), so it works regardless of what the frame contains -- no
    game-specific assumptions. Does not touch the real env."""
    start = env.clone()
    key0 = _masked_bytes(start.frame(), mask)
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
            key = _masked_bytes(c.frame(), mask)
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


def solve_by_search(env, actions=(1, 2, 3, 4), max_states=8000, mask=None):
    """Compose find_winning_path + execute_path: search on clones for a
    sequence that raises levels_completed, then replay it on the real env.
    Good default for any level whose winning state is reachable within a
    small (thousands of frames) BFS horizon. Pass `mask=detect_noise_mask(env)`
    for levels whose frame carries a HUD/counter that drifts every step
    regardless of action, so raw-frame dedup would never revisit a state."""
    path = find_winning_path(env, actions=actions, max_states=max_states, mask=mask)
    if path:
        execute_path(env, path)
    return path

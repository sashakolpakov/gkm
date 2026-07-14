# Shared leg library: small, named, reusable skills.
# Players import from here; add a NEW leg only when no existing leg fits.
#
# ---- Substrate facts discovered for g50t (raw frames, no game source) ----
#   * The avatar is a compact solid block; movement actions 1/2/3/4 translate it
#     (UP/DOWN/LEFT/RIGHT) and are blocked by non-floor cells.  Action 5 (USE)
#     acts on the tile the avatar stands on: on a "switch" tile it reconfigures
#     the maze (opens gates -> the avatar's reachable set grows); elsewhere it is
#     effectively a reset.
#   * A level is cleared (levels_completed increments) when the avatar reaches a
#     goal region.
#   * IMPORTANT: the environment carries HIDDEN state -- two states with byte
#     identical frames can transition differently.  So NEVER dedup a search on
#     raw frame bytes.  Planning is done on clones as concrete action sequences
#     (which ARE deterministic when replayed), deduping only on the avatar's
#     position within a fixed gate configuration.
from collections import deque

import numpy as np

MOVES = (1, 2, 3, 4)


# --------------------------------------------------------------------------
# Perception: locate the avatar generically by seeing which blob moves.
# --------------------------------------------------------------------------
def _components(mask):
    """4-connected components of a boolean mask -> list of (top_row, top_col, cells)."""
    seen = np.zeros(mask.shape, bool)
    rows, cols = mask.shape
    out = []
    for r in range(rows):
        for c in range(cols):
            if mask[r, c] and not seen[r, c]:
                stack = [(r, c)]
                seen[r, c] = True
                cells = []
                while stack:
                    x, y = stack.pop()
                    cells.append((x, y))
                    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < rows and 0 <= ny < cols and mask[nx, ny] and not seen[nx, ny]:
                            seen[nx, ny] = True
                            stack.append((nx, ny))
                out.append((min(p[0] for p in cells), min(p[1] for p in cells), cells))
    return out


def _shape_comps(frame, bg):
    """All non-background components as (color, shape_sig, top_left, cells)."""
    out = []
    for col in np.unique(frame):
        if col == bg:
            continue
        for tr, tc, cells in _components(frame == col):
            sig = frozenset((x - tr, y - tc) for x, y in cells)
            out.append((int(col), sig, (tr, tc), cells))
    return out


def avatar_tl(env):
    """Top-left (row, col) of the avatar in env's current frame.  The avatar is
    the object that TRANSLATES (same colour & shape, new position) when a move is
    applied.  This ignores flickering counters (shape changes, no match) and
    static scenery (position unchanged).  Returns None if nothing translates."""
    before = np.asarray(env.frame())
    vals, cnts = np.unique(before, return_counts=True)
    bg = int(vals[int(np.argmax(cnts))])
    bcomps = _shape_comps(before, bg)
    for a in MOVES:
        c = env.clone()
        c.step(a)
        after = np.asarray(c.frame())
        if np.array_equal(before, after):
            continue
        changed = before != after
        after_shapes = {(col, sig, tl) for col, sig, tl, _ in _shape_comps(after, bg)}
        for col, sig, tl, cells in bcomps:
            if not any(changed[x, y] for x, y in cells):
                continue
            # A translated copy: same colour+shape somewhere else in `after`.
            for col2, sig2, tl2 in after_shapes:
                if col2 == col and sig2 == sig and tl2 != tl:
                    return tl
    return None


# --------------------------------------------------------------------------
# Movement search inside a FIXED gate configuration (deterministic, position
# dedup is valid here).  Returns a reward-path if reachable, else the map of
# reachable positions -> move-path.
# --------------------------------------------------------------------------
def _move_explore(start_env):
    base = int(start_env.levels_completed)
    s = avatar_tl(start_env)
    if s is None:
        return None, {}
    seen = {s}
    paths = {s: []}
    q = deque([(start_env.clone(), s, [])])
    while q:
        node, pos, path = q.popleft()
        for a in MOVES:
            c = node.clone()
            c.step(a)
            if int(c.levels_completed) > base:
                return path + [a], paths          # reward reached by moving
            p = avatar_tl(c)
            if p is None or p == pos or p in seen:
                continue
            seen.add(p)
            paths[p] = path + [a]
            q.append((c, p, path + [a]))
    return None, paths


# --------------------------------------------------------------------------
# High level: reach the goal, unlocking gates with USE when plain movement is
# not enough.  Bounded number of switch presses.
# --------------------------------------------------------------------------
def plan_unlock_reach(env, max_toggles=2):
    """Return a concrete action list that raises levels_completed, or None.

    Strategy: try to reach the reward by movement alone; if not, try pressing
    USE at each reachable position (a candidate switch) and recurse with one
    fewer toggle.  A USE that opens gates enlarges the reachable set on the next
    round, letting movement finish the job.
    """
    def rec(node, toggles_left, visited_sizes):
        reward_path, reach = _move_explore(node)
        if reward_path is not None:
            return reward_path
        if toggles_left <= 0:
            return None
        # Prefer switches that CHANGE reachability (open gates); fall back to all.
        cur_size = len(reach)
        ranked = sorted(reach.items(), key=lambda kv: len(kv[1]))
        for pos, path in ranked:
            c = node.clone()
            for a in path:
                c.step(a)
            c.step(5)  # USE at this position
            _, reach2 = _move_explore(c)
            if len(reach2) <= cur_size:
                continue  # this USE did not open anything new; skip
            sub = rec(c, toggles_left - 1, visited_sizes)
            if sub is not None:
                return path + [5] + sub
        return None

    return rec(env.clone(), max_toggles, set())


def run_path(env, path):
    """Commit a planned action list on the real env, stopping if it terminates."""
    for a in path:
        if env.terminal():
            break
        env.step(a)


def solve_unlock_reach(env, max_toggles=2):
    """Leg: plan on a clone with plan_unlock_reach, then commit it. Returns True
    if a winning plan was found and committed."""
    plan = plan_unlock_reach(env, max_toggles=max_toggles)
    if not plan:
        return False
    run_path(env, plan)
    return True

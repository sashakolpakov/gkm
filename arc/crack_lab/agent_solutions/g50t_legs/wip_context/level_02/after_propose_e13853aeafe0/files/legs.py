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
import heapq
import time

import numpy as np

MOVES = (1, 2, 3, 4)


# --------------------------------------------------------------------------
# Cloning: the substrate has hidden state, so we always plan on throwaway
# clones and only touch the real env when committing.  Almost every planning
# step is "take a fresh clone and replay these actions on it" -- capture that
# once here so callers stay one-liners.
# --------------------------------------------------------------------------
def clone_after(env, actions):
    """Return a fresh clone of `env` with `actions` (an action or iterable of
    actions) applied in order.  Never mutates `env`."""
    if isinstance(actions, int):
        actions = (actions,)
    c = env.clone()
    for a in actions:
        c.step(a)
    return c


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
        c = clone_after(env, a)
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
            c = clone_after(node, a)
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
            c = clone_after(node, path + [5])  # replay to switch, then USE
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


# --------------------------------------------------------------------------
# Higher-order leg: the recurring "plan on a clone, then commit" shape.  Every
# solve_* leg is a planner (env -> action-list-or-None) wrapped by this.  Given
# the hidden-state substrate, planning MUST happen on a clone and the resulting
# concrete action list is what we replay on the real env.
# --------------------------------------------------------------------------
def plan_and_commit(env, planner, **kwargs):
    """Run `planner` on a clone of `env`; if it yields a non-empty action list,
    commit it on the real `env` and return True.  Otherwise return False and
    leave `env` untouched."""
    plan = planner(clone_after(env, ()), **kwargs)
    if not plan:
        return False
    run_path(env, plan)
    return True


def solve_unlock_reach(env, max_toggles=2):
    """Leg: plan an unlock-then-reach route and commit it. Returns True if a
    winning plan was found and committed."""
    return plan_and_commit(env, plan_unlock_reach, max_toggles=max_toggles)


# --------------------------------------------------------------------------
# Fast avatar position + reachability (no per-position re-cloning).
#
# `_move_explore`/`avatar_tl` call clone-heavy translation detection at every
# node, which is too slow for the multi-USE search needed on later levels.
# `_avatar_pos` reads the avatar straight from a frame (the largest compact
# color-9 blob that is not full-width scenery), and `fast_reach` does a single
# clone per (position,action) BFS edge, deduping on avatar position.  This is
# valid within a FIXED gate configuration (movement is deterministic there).
# --------------------------------------------------------------------------
def _avatar_pos(frame):
    """Top-left (row, col) of the avatar = largest compact color-9 blob that is
    not a full-width border line.  Returns None if none found."""
    f = np.asarray(frame)
    cand = [(tr, tc, cells) for tr, tc, cells in _components(f == 9)
            if 5 < len(cells) < 40 and tr < f.shape[0] - 4]
    if not cand:
        return None
    cand.sort(key=lambda c: -len(c[2]))
    return (cand[0][0], cand[0][1])


def fast_reach(start):
    """BFS over avatar POSITIONS inside the current (fixed) gate configuration.
    One clone per explored edge; dedup on avatar position.  Returns
    (reward_path, {pos: move_path}); reward_path is a winning move list if the
    reward can be raised by pure movement, else None."""
    base = int(start.levels_completed)
    s = _avatar_pos(start.frame())
    if s is None:
        return None, {}
    paths = {s: []}
    seen = {s}
    q = deque([(start.clone(), s, [])])
    while q:
        node, pos, path = q.popleft()
        for a in MOVES:
            c = node.clone()
            c.step(a)
            if int(c.levels_completed) > base:
                return path + [a], paths
            p = _avatar_pos(c.frame())
            if p is None or p == pos or p in seen:
                continue
            seen.add(p)
            paths[p] = path + [a]
            q.append((c, p, path + [a]))
    return None, paths


# --------------------------------------------------------------------------
# General unlock-then-reach via USE-MACROS.
#
# On these levels a USE performed after walking somewhere resets the avatar to
# its start BUT PRESERVES gates that were opened along the way (touching a wall
# segment can open a remote gate).  Chaining such "walk-somewhere-then-USE"
# macros accumulates opened gates until the goal becomes reachable by plain
# movement.  We search best-first over the resulting reachable-position sets:
# a macro = (walk to some reachable position P, press USE); the state key is the
# frozenset of currently reachable positions.  Unlike `plan_unlock_reach`, we do
# NOT require each individual USE to enlarge reachability -- an opening can take
# several staged macros before it pays off -- so this handles deeper unlock
# chains while staying deterministic and clone-bounded.
# --------------------------------------------------------------------------
def plan_unlock_macro(env, max_expand=400, time_limit=550):
    """Return a concrete winning action list, or None.  Plans on clones only."""
    t0 = time.time()
    rp0, reach0 = fast_reach(env)
    if rp0 is not None:
        return rp0
    seen = {frozenset(reach0)}
    # priority queue keyed by -reachable-count (explore the most-open configs
    # first); cnt is a monotone tiebreaker so ordering stays deterministic.
    pq = [(-len(reach0), 0, env.clone(), [])]
    cnt = 1
    while pq and cnt < max_expand and (time.time() - t0) < time_limit:
        _, _, node, prefix = heapq.heappop(pq)
        rp, reach = fast_reach(node)
        if rp is not None:
            return prefix + rp
        for pos, path in sorted(reach.items(), key=lambda kv: len(kv[1])):
            c = clone_after(node, path + [5])          # walk to P, then USE
            rp2, reach2 = fast_reach(c)
            if rp2 is not None:
                return prefix + path + [5] + rp2
            rk = frozenset(reach2)
            if rk in seen:
                continue
            seen.add(rk)
            heapq.heappush(pq, (-len(reach2), cnt, c, prefix + path + [5]))
            cnt += 1
    return None


def solve_unlock_macro(env, max_expand=400, time_limit=550):
    """Leg: plan a USE-macro unlock-then-reach route and commit it.  Returns
    True iff a winning plan was found and committed."""
    return plan_and_commit(env, plan_unlock_macro,
                           max_expand=max_expand, time_limit=time_limit)

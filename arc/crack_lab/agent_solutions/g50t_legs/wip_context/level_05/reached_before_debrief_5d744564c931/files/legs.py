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
#
# This is ONE skill -- "flood the avatar's positions with moves, one clone per
# edge, dedup on position" -- parameterised by HOW we locate the avatar in a
# clone (`locate: env -> pos | None`).  `_move_explore` uses the robust but
# clone-heavy translation detector; `fast_reach` uses the clone-cheap
# color-9-blob reader.  The BFS itself is written exactly once here.
# --------------------------------------------------------------------------
def _reach_bfs(start, locate):
    """BFS over avatar POSITIONS in the current (fixed) gate configuration.
    One clone per explored edge; dedup on the position returned by `locate`.
    Returns (reward_path, {pos: move_path}); reward_path is a winning move list
    if the reward can be raised by pure movement, else None."""
    base = int(start.levels_completed)
    s = locate(start)
    if s is None:
        return None, {}
    seen = {s}
    paths = {s: []}
    q = deque([(start.clone(), s, [])])
    while q:
        node, pos, path = q.popleft()
        for a in MOVES:
            c = clone_after(node, a)
            if int(c.levels_completed) > base:
                return path + [a], paths          # reward reached by moving
            p = locate(c)
            if p is None or p == pos or p in seen:
                continue
            seen.add(p)
            paths[p] = path + [a]
            q.append((c, p, path + [a]))
    return None, paths


def _move_explore(start_env):
    """Reach-BFS using the translation-based avatar detector (robust, slow)."""
    return _reach_bfs(start_env, avatar_tl)


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
    def rec(node, toggles_left):
        reward_path, reach = _move_explore(node)
        if reward_path is not None:
            return reward_path
        if toggles_left <= 0:
            return None
        # Only keep switches that CHANGE reachability (open new gates); a USE
        # that grows nothing is pruned so the bounded DFS stays shallow.
        cur_size = len(reach)
        for macro, c, _rp, reach2 in _use_macros(node, reach, _move_explore):
            if len(reach2) <= cur_size:
                continue  # this USE did not open anything new; skip
            sub = rec(c, toggles_left - 1)
            if sub is not None:
                return macro + sub
        return None

    return rec(env.clone(), max_toggles)


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
    """Reach-BFS using the clone-cheap color-9-blob avatar reader (fast)."""
    return _reach_bfs(start, lambda e: _avatar_pos(e.frame()))


# --------------------------------------------------------------------------
# The shared "unlock macro" expansion.  Both unlock planners advance the gate
# configuration the same way: from a node, pick a currently-reachable tile,
# walk there and press USE (a "macro"), yielding a child gate-config.  Cheapest
# walks are tried first so plans stay short.  Written ONCE here; the two
# planners differ only in how they SCHEDULE / PRUNE these successors.
# --------------------------------------------------------------------------
def _use_macros(node, reach, reach_fn):
    """Yield (macro, child_env, child_reward_path, child_reach) for every
    "walk to a reachable tile, then USE" macro, cheapest walk first.
      * macro              = walk move-path + [5] (USE)
      * child_env          = clone of `node` after that macro
      * child_reward_path  = winning move list from the child, or None
      * child_reach        = child's reachable-position map
    `reach_fn` is the reach-BFS to probe children with (must match `reach`)."""
    for _pos, path in sorted(reach.items(), key=lambda kv: len(kv[1])):
        macro = path + [5]
        c = clone_after(node, macro)
        rp2, reach2 = reach_fn(c)
        yield macro, c, rp2, reach2


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
        for macro, c, rp2, reach2 in _use_macros(node, reach, fast_reach):
            if rp2 is not None:
                return prefix + macro + rp2
            rk = frozenset(reach2)
            if rk in seen:                             # same gate config seen
                continue
            seen.add(rk)
            heapq.heappush(pq, (-len(reach2), cnt, c, prefix + macro))
            cnt += 1
    return None


def solve_unlock_macro(env, max_expand=400, time_limit=550):
    """Leg: plan a USE-macro unlock-then-reach route and commit it.  Returns
    True iff a winning plan was found and committed."""
    return plan_and_commit(env, plan_unlock_macro,
                           max_expand=max_expand, time_limit=time_limit)


# --------------------------------------------------------------------------
# Special-surface frontier unlock.
#
# Some configurations need a commit whose hidden effect is only exposed by a
# later commit: the first commit need not enlarge the avatar's reachable set.
# Searching every reachable tile makes that memory step needlessly broad.
# The persistent mechanisms are visibly colored surfaces, so restrict USE
# macros to avatar placements overlapping a non-UI/non-wall color.  Reachable
# position count is the dense progress measure; a bounded structural-change
# fallback bridges intermediate commits with no immediate reach growth.
# --------------------------------------------------------------------------
def _special_frontier(reach, frame):
    """Reach entries whose 5x5 avatar footprint overlaps a special surface."""
    f = np.asarray(frame)
    special = ~np.isin(f, (0, 1, 2, 5, 9))
    out = []
    for pos, path in reach.items():
        r, c = pos
        if special[r:r + 5, c:c + 5].any():
            out.append((pos, path))
    return sorted(out, key=lambda item: (len(item[1]), item[0]))


def plan_frontier_unlock(env, max_stages=10, max_stalls=2):
    """Plan staged USE commits by maximizing movement reachability.

    Candidate commits are limited to reachable special-colored surfaces.  A
    commit that grows the reachable-position set is always preferred.  When
    none grows it, the shortest commit producing the largest persistent world
    change is accepted, up to ``max_stalls`` times; this preserves hidden
    intermediate state without exploding into an unrestricted macro tree.
    """
    node = env.clone()
    prefix = []
    stalls = 0
    seen = set()

    for _ in range(max_stages):
        reward_path, reach = fast_reach(node)
        if reward_path is not None:
            return prefix + reward_path

        state_key = (frozenset(reach), np.asarray(node.frame())[7:-1].tobytes())
        if state_key in seen:
            return None
        seen.add(state_key)

        candidates = _special_frontier(reach, node.frame())
        if not candidates:
            candidates = sorted(reach.items(), key=lambda item: (len(item[1]), item[0]))

        growth = []
        changed = []
        base_world = np.asarray(node.frame())[7:-1]
        for pos, path in candidates:
            macro = path + [5]
            child = clone_after(node, macro)
            child_reward, child_reach = fast_reach(child)
            if child_reward is not None:
                return prefix + macro + child_reward

            item = (len(child_reach), -len(macro), pos, macro, child, child_reach)
            if len(child_reach) > len(reach):
                growth.append(item)
            else:
                delta = int(np.count_nonzero(
                    base_world != np.asarray(child.frame())[7:-1]))
                if delta:
                    changed.append((delta, -len(macro), pos, macro, child, child_reach))

        if growth:
            _, _, _, macro, node, _ = max(growth, key=lambda item: item[:3])
            prefix += macro
            stalls = 0
            continue
        if changed and stalls < max_stalls:
            _, _, _, macro, node, _ = max(changed, key=lambda item: item[:3])
            prefix += macro
            stalls += 1
            continue
        return None
    return None


def solve_frontier_unlock(env, max_stages=10, max_stalls=2):
    """Leg: plan and commit a special-surface frontier unlock route."""
    return plan_and_commit(env, plan_frontier_unlock,
                           max_stages=max_stages, max_stalls=max_stalls)

# Shared leg library: small, named, reusable skills.
# Players import from here; add a NEW leg only when no existing leg fits.
#
# tr87 mechanic (discovered by experiment on clones):
#   A row of "tiles" can each be cycled through a small fixed set of glyphs by
#   an EDIT action; a MOVE action advances a cursor to the next tile. The level
#   completes when every tile is set to a hidden target glyph. There is no
#   visible per-tile feedback, so we search the (small) reachable config space
#   of a clone for a config that raises levels_completed, then replay the path.
import time
import numpy as np
import perception as P


# --- generic observational primitives (written ONCE, reused everywhere) -----

def action_bbox(env, action):
    """Bounding box of the frame region that `action` changes, from `env`.

    Clones `env`, steps `action`, and returns the delta bbox
    (r0, c0, r1, c1) or None if the action changes nothing.
    """
    before = np.asarray(env.frame())
    c = env.clone()
    c.step(action)
    return P.frame_delta(before, c.frame())["bbox"]


def bbox_size(bbox):
    """(height, width) of an inclusive bbox, or (0, 0) if bbox is None."""
    if not bbox:
        return 0, 0
    r0, c0, r1, c1 = bbox
    return (r1 - r0 + 1), (c1 - c0 + 1)


def action_period(env, action, key_fn, max_k=16, default=None):
    """How many repeats of `action` until `key_fn(state)` returns to its start.

    Steps `action` on a clone, watching the observation `key_fn`. Returns the
    first k in 1..max_k-1 for which the observation equals its initial value
    (the period), else `default`.
    """
    c = env.clone()
    k0 = key_fn(c)
    for k in range(1, max_k):
        c.step(action)
        if key_fn(c) == k0:
            return k
    return default


def replay_for_reward(env, path):
    """Execute `path` on the real `env`; return True if levels_completed rose.

    Stops early on a terminal frame. This is the generic "commit the winning
    plan found on a clone" skill.
    """
    before = env.levels_completed
    for a in path:
        if env.terminal():
            break
        env.step(a)
    return env.levels_completed > before


def solve_by_clone_search(env, discover, search):
    """Higher-order leg: perceive -> search-on-clone -> replay-on-real.

    `discover(env)` returns an opaque puzzle `spec` (or None if it does not
    apply). `search(env, spec)` returns a reward-raising action path (or None).
    The winning path is then committed to the real env via replay_for_reward.
    Returns True iff a level was cleared.
    """
    spec = discover(env)
    if spec is None:
        return False
    path = search(env, spec)
    if not path:
        return False
    return replay_for_reward(env, path)


# --- tile-cycle puzzle (tr87) -----------------------------------------------

def discover_tile_cycle_puzzle(env):
    """Auto-discover the tile/cursor structure from action deltas on clones.

    Returns dict with keys: edit (action that cycles the current tile),
    move (action that advances the cursor to the next tile), n_tiles, cycle.
    Returns None if the frame does not look like a tile-cycle puzzle.
    """
    acts = list(env.actions)
    dbbox = {a: action_bbox(env, a) for a in acts}

    # EDIT actions produce a small, localized change (a single tile).
    def small(bb):
        h, w = bbox_size(bb)
        return bb and h <= 8 and w <= 8
    edits = [a for a in acts if small(dbbox[a])]
    moves = [a for a in acts if dbbox[a] and a not in edits]
    if not edits or not moves:
        return None
    edit = edits[0]
    move = min(moves, key=lambda a: bbox_size(dbbox[a])[1])  # forward = adjacent

    er0, ec0, er1, ec1 = dbbox[edit]

    # cursor position observation: the left column of the current edit region.
    def edit_left_col(c):
        bb = action_bbox(c, edit)
        return bb[1] if bb else None

    # glyph observation: the pixels of the current tile region.
    def tile_key(c):
        b = np.asarray(c.frame())
        return b[er0:er1 + 1, ec0:ec1 + 1].tobytes()

    # n_tiles: repeats of MOVE until the cursor wraps back to the start tile.
    n_tiles = action_period(env, move, edit_left_col, max_k=16, default=16)
    # cycle: repeats of EDIT until the tile glyph returns to its start.
    cycle = action_period(env, edit, tile_key, max_k=16, default=8)
    return {"edit": edit, "move": move, "n_tiles": n_tiles, "cycle": cycle}


def search_tile_cycle_config(env, edit, move, n_tiles, cycle, budget_s=200):
    """Nested-clone DFS over per-tile glyph states; early-exit on reward.

    Returns the action path (list of ints) that raises levels_completed, or None.
    """
    t0 = time.time()
    found = [None]

    def rec(clone, ti, path):
        if found[0] is not None or time.time() - t0 > budget_s:
            return
        if clone.levels_completed > 0:
            found[0] = path; return
        if ti == n_tiles:
            return
        if ti == n_tiles - 1:
            # Last tile: reuse one clone, stepping EDIT and checking reward.
            c = clone.clone()
            if c.levels_completed > 0:
                found[0] = path; return
            for s in range(1, cycle):
                c.step(edit)
                if c.levels_completed > 0:
                    found[0] = path + [edit] * s; return
            return
        for s in range(cycle):
            c = clone.clone()
            for _ in range(s):
                c.step(edit)
            if c.levels_completed > 0:
                found[0] = path + [edit] * s; return
            nc = c.clone(); nc.step(move)
            rec(nc, ti + 1, path + [edit] * s + [move])
            if found[0] is not None:
                return

    rec(env.clone(), 0, [])
    return found[0]


def solve_tile_cycle_puzzle(env, budget_s=200):
    """Thin composition: discover the tile-cycle structure, search a clone for
    a winning config, then replay that path on the real env (see
    solve_by_clone_search). Returns True iff a level was cleared.
    """
    return solve_by_clone_search(
        env,
        discover=discover_tile_cycle_puzzle,
        search=lambda e, spec: search_tile_cycle_config(
            e, spec["edit"], spec["move"], spec["n_tiles"], spec["cycle"],
            budget_s),
    )

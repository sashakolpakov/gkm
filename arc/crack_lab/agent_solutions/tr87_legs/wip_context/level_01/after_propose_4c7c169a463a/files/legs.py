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


def discover_tile_cycle_puzzle(env):
    """Auto-discover the tile/cursor structure from action deltas on clones.

    Returns dict with keys: edit (action that cycles the current tile),
    move (action that advances the cursor to the next tile), n_tiles, cycle.
    Returns None if the frame does not look like a tile-cycle puzzle.
    """
    acts = list(env.actions)
    base = np.asarray(env.frame())
    dbbox, dwidth = {}, {}
    for a in acts:
        c = env.clone(); c.step(a)
        d = P.frame_delta(base, c.frame()); bb = d["bbox"]
        dbbox[a] = bb
        dwidth[a] = (bb[3] - bb[1] + 1) if bb else 0

    # EDIT actions produce a small, localized change (a single tile).
    edits = [a for a in acts if dbbox[a]
             and (dbbox[a][3] - dbbox[a][1] + 1) <= 8
             and (dbbox[a][2] - dbbox[a][0] + 1) <= 8]
    moves = [a for a in acts if dbbox[a] and a not in edits]
    if not edits or not moves:
        return None
    edit = edits[0]
    move = min(moves, key=lambda a: dwidth[a])  # forward move = adjacent tile

    er0, ec0, er1, ec1 = dbbox[edit]

    def edit_left_col(c):
        b = np.asarray(c.frame()); cc = c.clone(); cc.step(edit)
        bb = P.frame_delta(b, cc.frame())["bbox"]
        return bb[1] if bb else None

    # n_tiles: advance the cursor with `move` until the edit region returns.
    cols = []
    c = env.clone()
    for _ in range(16):
        cols.append(edit_left_col(c)); c.step(move)
    n_tiles = next((k for k in range(1, len(cols)) if cols[k] == cols[0]), len(cols))

    # cycle length: repeatedly EDIT one tile until the tile region repeats.
    def tile_key(c):
        b = np.asarray(c.frame()); return b[er0:er1 + 1, ec0:ec1 + 1].tobytes()
    c = env.clone(); k0 = tile_key(c); cycle = None
    for k in range(1, 16):
        c.step(edit)
        if tile_key(c) == k0:
            cycle = k; break
    if cycle is None:
        cycle = 8
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
    """Discover the tile-cycle puzzle, search a clone for a winning config,
    then replay that path on the real env. Returns True if a level was cleared.
    """
    spec = discover_tile_cycle_puzzle(env)
    if spec is None:
        return False
    path = search_tile_cycle_config(env, spec["edit"], spec["move"],
                                    spec["n_tiles"], spec["cycle"], budget_s)
    if not path:
        return False
    before = env.levels_completed
    for a in path:
        if env.terminal():
            break
        env.step(a)
    return env.levels_completed > before

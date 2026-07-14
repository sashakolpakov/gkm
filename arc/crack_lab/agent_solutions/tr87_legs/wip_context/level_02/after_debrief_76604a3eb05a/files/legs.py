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


# --- tile-row editing primitive (shared by every tile-cycle solver) ---------

def tile_cycle_path(edit, move, counts):
    """Serialize a per-tile plan into an edit/move action path.

    `counts[i]` is how many EDIT presses tile i needs; tiles are visited left
    to right, a single MOVE separating each tile's edit-run from the next (no
    trailing move). This is the ONE place that turns a "what glyph does each
    tile want" plan into concrete actions, so every tile-row solver -- whether
    the target counts come from a clone search (L1) or from decoding the frame
    (L2) -- shares the exact same serialization.
    """
    path = []
    for ti, s in enumerate(counts):
        path += [edit] * s
        if ti < len(counts) - 1:
            path += [move]
    return path


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
    found = [None]  # holds the winning per-tile edit-count list

    def rec(clone, ti, counts):
        if found[0] is not None or time.time() - t0 > budget_s:
            return
        if clone.levels_completed > 0:
            found[0] = list(counts); return
        if ti == n_tiles:
            return
        if ti == n_tiles - 1:
            # Last tile: reuse one clone, stepping EDIT and checking reward.
            c = clone.clone()
            for s in range(cycle):
                if c.levels_completed > 0:
                    found[0] = counts + [s]; return
                c.step(edit)
            return
        for s in range(cycle):
            c = clone.clone()
            for _ in range(s):
                c.step(edit)
            if c.levels_completed > 0:
                found[0] = counts + [s]; return
            nc = c.clone(); nc.step(move)
            rec(nc, ti + 1, counts + [s])
            if found[0] is not None:
                return

    rec(env.clone(), 0, [])
    if found[0] is None:
        return None
    return tile_cycle_path(edit, move, found[0])


# --- glyph-cipher tile puzzle (tr87 L2: decode a legend, no reward feedback) --

def _glyph(frame, r, c, gh, gw, fg):
    """Binary (fg?1:0) gh x gw glyph read at top-left (r,c)."""
    return tuple(tuple(1 if frame[r + i, c + j] == fg else 0
                       for j in range(gw)) for i in range(gh))


def _glyph_canon(g):
    """Canonical form of a binary glyph under the 8 D4 symmetries.

    Matching by canonical form is orientation-invariant: two glyphs that are
    rotations/reflections of each other compare equal. This is what makes the
    legend lookup robust when the same symbol is drawn in different poses.
    """
    a = np.array(g)
    best = None
    cur = a
    for _ in range(4):
        for m in (cur, cur[:, ::-1]):
            t = tuple(map(tuple, m.tolist()))
            if best is None or t < best:
                best = t
        cur = np.rot90(cur)
    return best


def _bordered_boxes(frame, color, inner_h, pitch, min_area=15):
    """Connected components of `color` that are `inner_h`+2 tall (a glyph row
    framed by a 1-px border). Returns Blob-like objects with .bbox/.color."""
    blobs = P.connected_components(frame, min_area=min_area)
    return [b for b in blobs
            if b.color == color and (b.bbox[2] - b.bbox[0] + 1) == inner_h + 2]


def _box_glyphs(frame, bbox, gh, gw, pitch, fg):
    """Split a bordered box into its 1..k glyphs (pitch-spaced along columns)."""
    r0, c0, r1, c1 = bbox
    k = (c1 - c0 + 1) // pitch
    return [_glyph(frame, r0 + 1, c0 + 1 + pitch * g, gh, gw, fg) for g in range(k)]


def discover_glyph_cipher_puzzle(env):
    """Auto-discover a legend-decode tile puzzle from a single frame.

    Layout (all inferred, nothing hardcoded):
      * a row of editable glyph tiles cycled by an EDIT action, cursor advanced
        by a MOVE action (reuses discover_tile_cycle_puzzle);
      * small 'key' boxes (border color BOX, single glyph) each paired with a
        'value' region (border color VAL, 1..k glyphs) on the same rows to its
        right -> a legend mapping key-symbol -> value-glyph-sequence;
      * one wide 'target' box (border BOX, multiple glyphs): the coded word.
    Decoding the target word through the legend yields the sequence of glyphs
    the editable tiles must be set to. Returns a spec dict or None.
    """
    spec = discover_tile_cycle_puzzle(env)
    if spec is None:
        return None
    edit, move = spec["edit"], spec["move"]
    n_tiles, cycle = spec["n_tiles"], spec["cycle"]
    f = np.asarray(env.frame())
    ebb = action_bbox(env, edit)
    if not ebb:
        return None
    r0, c0, r1, c1 = ebb
    gh, gw = r1 - r0 + 1, c1 - c0 + 1
    pitch = gw + 2

    # VAL border color: ring around the edit region; FG: the non-VAL color in it.
    from collections import Counter
    ring = []
    for c in range(c0 - 1, c1 + 2):
        ring.append(int(f[r0 - 1, c])); ring.append(int(f[r1 + 1, c]))
    for r in range(r0 - 1, r1 + 2):
        ring.append(int(f[r, c0 - 1])); ring.append(int(f[r, c1 + 1]))
    VAL = Counter(ring).most_common(1)[0][0]
    inside = f[r0:r1 + 1, c0:c1 + 1]
    fgc = [int(v) for v in np.unique(inside) if int(v) != VAL]
    FG = fgc[0] if fgc else 5

    # BOX color: the other bordered-box color present (glyph-tall components).
    blobs = P.connected_components(f, min_area=15)
    tall = [b for b in blobs if (b.bbox[2] - b.bbox[0] + 1) == gh + 2]
    boxcolors = Counter(b.color for b in tall if b.color != VAL)
    if not boxcolors:
        return None
    BOX = boxcolors.most_common(1)[0][0]

    valregs = [b for b in tall if b.color == VAL]
    boxes = [b for b in tall if b.color == BOX]

    def contains(b, bb):
        R0, C0, R1, C1 = b.bbox
        return R0 <= bb[0] and R1 >= bb[2] and C0 <= bb[1] and C1 >= bb[3]

    editregs = [b for b in valregs if contains(b, ebb)]
    if not editregs:
        return None
    editreg = editregs[0]
    legvals = [b for b in valregs if b is not editreg]
    keyboxes = [b for b in boxes if (b.bbox[3] - b.bbox[1] + 1) == pitch]
    targetboxes = [b for b in boxes if (b.bbox[3] - b.bbox[1] + 1) > pitch]
    if not keyboxes or not targetboxes:
        return None
    tb = targetboxes[0]

    keys = [_box_glyphs(f, b.bbox, gh, gw, pitch, FG)[0] for b in keyboxes]
    keycanon = [_glyph_canon(k) for k in keys]

    def pairval(kb):
        R0, C0, R1, C1 = kb.bbox
        cands = [v for v in legvals if v.bbox[0] == R0 and v.bbox[1] > C1]
        cands.sort(key=lambda v: v.bbox[1] - C1)
        return cands[0] if cands else None

    keyvals = []
    for kb in keyboxes:
        pv = pairval(kb)
        if pv is None:
            return None
        keyvals.append(_box_glyphs(f, pv.bbox, gh, gw, pitch, FG))

    # decode the target word
    targets = _box_glyphs(f, tb.bbox, gh, gw, pitch, FG)
    seq = []
    for t in targets:
        ct = _glyph_canon(t)
        ki = [i for i, kc in enumerate(keycanon) if kc == ct]
        if len(ki) != 1:
            return None
        seq.extend(keyvals[ki[0]])
    if len(seq) != n_tiles:
        return None

    return {"edit": edit, "move": move, "n_tiles": n_tiles, "cycle": cycle,
            "gh": gh, "gw": gw, "pitch": pitch, "fg": FG,
            "editreg": editreg.bbox, "seq_canon": [_glyph_canon(g) for g in seq]}


def plan_glyph_cipher(env, spec):
    """Compute the edit/move path that sets each tile to its decoded target.

    Each tile is cycled (on a clone) until its glyph matches the required
    canonical target; the edit-count is recorded. Returns the action path
    (list of ints) or None if any tile cannot reach its target.
    """
    edit, move, cycle = spec["edit"], spec["move"], spec["cycle"]
    gh, gw, pitch, fg = spec["gh"], spec["gw"], spec["pitch"], spec["fg"]
    R0, C0, R1, C1 = spec["editreg"]
    seq_canon = spec["seq_canon"]
    n = spec["n_tiles"]
    counts = []
    for ti in range(n):
        c = env.clone()
        for _ in range(ti):
            c.step(move)
        found = None
        for s in range(cycle):
            tg = _glyph(np.asarray(c.frame()), R0 + 1, C0 + 1 + pitch * ti, gh, gw, fg)
            if _glyph_canon(tg) == seq_canon[ti]:
                found = s
                break
            c.step(edit)
        if found is None:
            return None
        counts.append(found)
    return tile_cycle_path(edit, move, counts)


def solve_glyph_cipher_puzzle(env):
    """Discover -> decode -> plan -> replay a legend-decode tile puzzle.

    Returns True iff a level was cleared."""
    return solve_by_clone_search(
        env,
        discover=discover_glyph_cipher_puzzle,
        search=plan_glyph_cipher,
    )


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

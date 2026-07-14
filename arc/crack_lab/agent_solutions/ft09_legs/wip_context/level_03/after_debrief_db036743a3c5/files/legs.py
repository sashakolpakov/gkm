# Shared leg library: small, named, reusable skills.
# Players import from here; add a NEW leg only when no existing leg fits.
import itertools
from collections import Counter

import numpy as np


def _changed_cells(base, frame, row_limit=62):
    """Cells that differ between two frames, ignoring a bottom status bar."""
    d = np.argwhere(np.asarray(base) != np.asarray(frame))
    return [(int(r), int(c)) for r, c in d if r < row_limit]


def discover_toggle_tiles(env, step=2, row_limit=62):
    """Probe coordinate clicks on a clone and return one representative (x, y)
    click per independent toggle region. A "tile" is identified by the bounding
    box of the cells it changes (excluding the bottom status bar). Generic to any
    coordinate-click board where clicking a cell mutates a bounded region.
    """
    base = np.asarray(env.frame()).copy()
    tiles = {}
    for y in range(0, base.shape[0], step):
        for x in range(0, base.shape[1], step):
            c = env.clone()
            c.step(6, x, y)
            cells = _changed_cells(base, c.frame(), row_limit)
            if not cells:
                continue
            rs = [p[0] for p in cells]
            cs = [p[1] for p in cells]
            key = (min(rs), min(cs), max(rs), max(cs))
            # keep the click closest to the region center as representative
            cx, cy = (min(cs) + max(cs)) // 2, (min(rs) + max(rs)) // 2
            prev = tiles.get(key)
            if prev is None:
                tiles[key] = (x, y)
            else:
                px, py = prev
                if (x - cx) ** 2 + (y - cy) ** 2 < (px - cx) ** 2 + (py - cy) ** 2:
                    tiles[key] = (x, y)
    return [tiles[k] for k in sorted(tiles)]


def search_toggle_solution(env, tiles=None, max_tiles=16):
    """Find a subset of tile clicks (each tile toggled at most once) that raises
    levels_completed, searching on clones from fewest clicks upward. Returns the
    click list [(x, y), ...] or None. Assumes independent toggle tiles whose
    target configuration is reached by clicking the right subset once each.
    """
    if tiles is None:
        tiles = discover_toggle_tiles(env)
    if not tiles or len(tiles) > max_tiles:
        return None
    base_level = env.levels_completed
    n = len(tiles)
    for k in range(0, n + 1):
        for combo in itertools.combinations(range(n), k):
            c = env.clone()
            for i in combo:
                x, y = tiles[i]
                c.step(6, x, y)
            if c.levels_completed > base_level:
                return [tiles[i] for i in combo]
    return None


def commit_plan(env, plan, apply_fn):
    """Replay a planned action sequence on the real env, one action at a time.
    `apply_fn(env, a)` executes a single planned action `a`. Returns True."""
    for a in plan:
        apply_fn(env, a)
    return True


def plan_and_commit(env, planner, apply_fn):
    """Higher-order leg: search for a plan on clones, then commit it for real.

    `planner(env)` searches (typically on env.clone()s) and returns an action
    sequence that reaches the goal, or a falsy value if none is found.
    `apply_fn(env, a)` executes one planned action on the real env. This is the
    recurring "plan on clones -> replay on real env" composition; write the
    commit loop ONCE here and pass in the planner + action applier.
    Returns True iff a plan was found and committed.
    """
    plan = planner(env)
    if not plan:
        return False
    return commit_plan(env, plan, apply_fn)


def _click_xy(env, xy):
    """Apply a single coordinate click (x, y) via the ARC click action."""
    x, y = xy
    env.step(6, x, y)


def solve_click_board(env, planner):
    """Higher-order leg for coordinate-click boards: plan a set of (x, y) clicks
    with `planner` (on clones), then commit them on the real env.

    This is `plan_and_commit` with the coordinate-click applier `_click_xy` bound
    ONCE. Every ft09 board so far reduces to "produce a click list, then click".
    The ONLY per-board part is the planner, which returns the goal-reaching click
    list [(x, y), ...] (or a falsy value): a subset SEARCH on clones
    (search_toggle_solution) or a pixel DECODE (discover_pattern_key_clicks).
    Returns True iff a plan was found and committed."""
    return plan_and_commit(env, planner, _click_xy)


def solve_toggle_board(env):
    """Discover the toggle board, search for a completing click subset, and
    commit those clicks on the real env. Returns True on success. Thin
    composition of solve_click_board over the subset-search toggle planner."""
    return solve_click_board(env, search_toggle_solution)


def _grid_of_blocks(frame, row_limit=62):
    """Perceive a regular grid of solid square blocks over a background.

    Returns (block_size, bg, slots, uniform, patterns) where `slots` maps a grid
    index (i, j) -> (top_row, left_col), `uniform` is the set of indices whose
    block is a single solid colour, and `patterns` maps index -> the block's
    pixel array for the non-uniform (decorated) blocks. Block size is auto
    detected from the small candidate set {4, 6, 8}. Generic to any board drawn
    as equal square cells on a lattice (e.g. the ft09 pattern-key boards).
    """
    f = np.asarray(frame)
    core = f[:row_limit]
    bg = Counter(core.ravel().tolist()).most_common(1)[0][0]
    nonbg = core != bg
    H, W = core.shape
    # Block size = modal length of maximal non-background horizontal runs.
    runs = Counter()
    for r in range(H):
        run = 0
        for c in range(W):
            if nonbg[r, c]:
                run += 1
            else:
                if run:
                    runs[run] += 1
                run = 0
        if run:
            runs[run] += 1
    if not runs:
        return None
    s = runs.most_common(1)[0][0]
    starts = []
    for r in range(0, H - s + 1):
        for c in range(0, W - s + 1):
            if nonbg[r:r + s, c:c + s].all():
                if (r == 0 or not nonbg[r - 1, c]) and (c == 0 or not nonbg[r, c - 1]):
                    starts.append((r, c))
    if not starts:
        return None
    rows = sorted({r for r, _ in starts})
    cols = sorted({c for _, c in starts})
    ri = {r: i for i, r in enumerate(rows)}
    ci = {c: j for j, c in enumerate(cols)}
    slots, uniform, patterns = {}, set(), {}
    for (r, c) in starts:
        blk = core[r:r + s, c:c + s]
        idx = (ri[r], ci[c])
        slots[idx] = (r, c)
        if len(np.unique(blk)) == 1:
            uniform.add(idx)
        else:
            patterns[idx] = blk
    return s, bg, slots, uniform, patterns


def discover_pattern_key_clicks(env):
    """Planner for the ft09 'pattern-key' toggle board.

    The board is a lattice of solid square blocks; a few 'pattern' blocks carry a
    3x3 mini-key (a solid centre colour plus mark/blank cells around it). Each
    pattern block tells which of its 8 grid-neighbour blocks to toggle: a
    neighbour is selected iff  (centre == block_colour) == (cell == mark_colour).
    Returns the list of (x, y) clicks (block centres) that reach the target, or
    None if the board is not a pattern-key board. Pure perception -- no search.
    """
    g = _grid_of_blocks(env.frame())
    if not g:
        return None
    s, bg, slots, uniform, patterns = g
    if not patterns or not uniform:
        return None
    block_color = Counter(
        int(np.asarray(env.frame())[r + s // 2, c + s // 2])
        for (r, c) in (slots[i] for i in uniform)
    ).most_common(1)[0][0]

    def cell3(blk):
        n = blk.shape[0]
        off = [i * n // 3 + n // 6 for i in range(3)]
        return [[int(blk[off[i], off[j]]) for j in range(3)] for i in range(3)]

    cellvals = Counter()
    for blk in patterns.values():
        gg = cell3(blk)
        for i in range(3):
            for j in range(3):
                if (i, j) != (1, 1):
                    cellvals[gg[i][j]] += 1
    common = cellvals.most_common()
    mark = common[-1][0] if len(common) > 1 else common[0][0]

    sel = set()
    for (pi, pj), blk in patterns.items():
        gg = cell3(blk)
        center = gg[1][1]
        for i in range(3):
            for j in range(3):
                if (i, j) == (1, 1):
                    continue
                nb = (pi + (i - 1), pj + (j - 1))
                if nb in uniform and ((center == block_color) == (gg[i][j] == mark)):
                    sel.add(nb)
    clicks = []
    for idx in sorted(sel):
        r, c = slots[idx]
        clicks.append((c + s // 2 - 1, r + s // 2 - 1))
    return clicks or None


def solve_pattern_key_board(env):
    """Decode the pattern-key board's target from the frame and commit the
    completing block toggles on the real env. Thin composition of solve_click_board
    over the pixel-decode pattern-key planner."""
    return solve_click_board(env, discover_pattern_key_clicks)

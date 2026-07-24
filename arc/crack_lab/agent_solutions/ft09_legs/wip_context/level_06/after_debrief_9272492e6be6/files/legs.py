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
    n = len(tiles)
    for k in range(0, n + 1):
        for combo in itertools.combinations(range(n), k):
            clicks = [tiles[i] for i in combo]
            if plan_reaches_next_level(env, clicks, _click_xy):
                return clicks
    return None


def commit_plan(env, plan, apply_fn):
    """Replay a planned action sequence on the real env, one action at a time.
    `apply_fn(env, a)` executes a single planned action `a`. Returns True."""
    for a in plan:
        apply_fn(env, a)
    return True


def plan_reaches_next_level(env, plan, apply_fn):
    """Test a plan on a clone and report whether it advances one level."""
    clone = env.clone()
    base_level = env.levels_completed
    commit_plan(clone, plan, apply_fn)
    return clone.levels_completed > base_level


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


def pattern_key_grid(env):
    """Perceive a pattern-key board and enforce its shape in ONE place.

    A pattern-key board is a lattice of solid square blocks (via _grid_of_blocks)
    carrying at least one decorated 'pattern' block AND at least one plain
    'uniform' block. Returns the grid tuple (s, bg, slots, uniform, patterns) or
    None when the frame is not a pattern-key board. Both pattern-key decoders
    share this perception + guard preamble."""
    g = _grid_of_blocks(env.frame())
    if not g:
        return None
    _, _, _, uniform, patterns = g
    if not patterns or not uniform:
        return None
    return g


def pattern_key_context(env):
    """Return the shared, validated perception context for a key-board planner.

    The three pattern-key variants all begin by recognizing the same lattice and
    taking the same frame snapshot. Keep that setup in one leg so each planner
    contains only its variant-specific decode rule.
    """
    g = pattern_key_grid(env)
    if not g:
        return None
    s, _, slots, uniform, patterns = g
    return s, slots, uniform, patterns, np.asarray(env.frame())


def read_mini_key(blk):
    """Read a decorated block's 3x3 mini-key: sample the centre pixel of each of
    the nine sub-cells and return a 3x3 list of colour values. The mini-key's
    centre (gg[1][1]) is the key colour; the surrounding cells are mark/blank.
    Shared by every pattern-key decoder."""
    n = blk.shape[0]
    off = [i * n // 3 + n // 6 for i in range(3)]
    return [[int(blk[off[i], off[j]]) for j in range(3)] for i in range(3)]


def mark_color(patterns):
    """The 'mark' colour of a set of mini-keys: the least-common non-centre cell
    value across all decorated blocks (marks are sparse, blanks are common).
    Shared by the pattern-key decoders."""
    cellvals = Counter()
    for blk in patterns.values():
        gg = read_mini_key(blk)
        for i in range(3):
            for j in range(3):
                if (i, j) != (1, 1):
                    cellvals[gg[i][j]] += 1
    return cellvals.most_common()[-1][0]


def block_click_xy(slots, idx, s):
    """The coordinate-click (x, y) that targets the centre of block `idx` on a
    lattice of size-`s` cells. Shared by the pattern-key decoders."""
    r, c = slots[idx]
    return (c + s // 2 - 1, r + s // 2 - 1)


def block_center_color(frame, slots, idx, s):
    """The solid colour currently shown by block `idx` (its centre pixel)."""
    r, c = slots[idx]
    return int(np.asarray(frame)[r + s // 2, c + s // 2])


def key_marked_neighbours(patterns, uniform, mark):
    """Yield (neighbour_idx, key_centre_colour) for every marked neighbour of
    every mini-key: a mark cell in a pattern block points at the grid neighbour
    in that direction. Only neighbours that are plain uniform blocks are yielded.
    This is the shared "read the marks" skeleton of the pattern-key decoders."""
    for (pi, pj), blk in patterns.items():
        gg = read_mini_key(blk)
        center = gg[1][1]
        for i in range(3):
            for j in range(3):
                if (i, j) == (1, 1):
                    continue
                if gg[i][j] == mark:
                    nb = (pi + (i - 1), pj + (j - 1))
                    if nb in uniform:
                        yield nb, center


def discover_pattern_key_clicks(env):
    """Planner for the ft09 'pattern-key' toggle board.

    The board is a lattice of solid square blocks; a few 'pattern' blocks carry a
    3x3 mini-key (a solid centre colour plus mark/blank cells around it). Each
    pattern block tells which of its 8 grid-neighbour blocks to toggle: a
    neighbour is selected iff  (centre == block_colour) == (cell == mark_colour).
    Returns the list of (x, y) clicks (block centres) that reach the target, or
    None if the board is not a pattern-key board. Pure perception -- no search.
    """
    context = pattern_key_context(env)
    if not context:
        return None
    s, slots, uniform, patterns, f = context
    block_color = Counter(
        block_center_color(f, slots, i, s) for i in uniform
    ).most_common(1)[0][0]
    mark = mark_color(patterns)

    # 2-state toggle rule: a grid-neighbour block is selected iff
    # (key centre == block colour) == (cell == mark colour).
    sel = set()
    for (pi, pj), blk in patterns.items():
        gg = read_mini_key(blk)
        center = gg[1][1]
        for i in range(3):
            for j in range(3):
                if (i, j) == (1, 1):
                    continue
                nb = (pi + (i - 1), pj + (j - 1))
                if nb in uniform and ((center == block_color) == (gg[i][j] == mark)):
                    sel.add(nb)
    clicks = [block_click_xy(slots, idx, s) for idx in sorted(sel)]
    return clicks or None


def solve_pattern_key_board(env):
    """Decode the pattern-key board's target from the frame and commit the
    completing block toggles on the real env. Thin composition of solve_click_board
    over the pixel-decode pattern-key planner."""
    return solve_click_board(env, discover_pattern_key_clicks)


def discover_multistate_key_clicks(env):
    """Planner for a MULTI-STATE pattern-key toggle board.

    Generalises discover_pattern_key_clicks to cells that CYCLE through k states
    on each click (e.g. 9->8->12->9). The board is a lattice of solid square
    blocks (perceived by _grid_of_blocks); a few 'pattern' blocks carry a 3x3
    mini-key: a solid centre colour plus mark/blank cells. Rule:

      * each pattern's marked neighbour block has TARGET colour == that pattern's
        CENTRE colour (marks and centres are read straight from the mini-key);
      * every other (unmarked) uniform block shares one DEFAULT target colour.

    The click cost to drive a block from its current colour to its target is the
    forward distance in the click cycle (discovered by probing one block on a
    clone). The single unknown DEFAULT is resolved by trying each cycle colour on
    a clone and keeping the one that raises levels_completed. Returns the full
    click list [(x, y), ...] (block centres, repeated per required click) or None.
    """
    context = pattern_key_context(env)
    if not context:
        return None
    s, slots, uniform, patterns, f = context

    # Discover the click cycle by probing one uniform block on a clone.
    idx0 = next(iter(uniform))
    px, py = block_click_xy(slots, idx0, s)
    cl = env.clone()
    cycle = [block_center_color(cl.frame(), slots, idx0, s)]
    for _ in range(8):
        cl.step(6, px, py)
        col = block_center_color(cl.frame(), slots, idx0, s)
        if col == cycle[0]:
            break
        cycle.append(col)
    ncy = len(cycle)
    pos = {col: i for i, col in enumerate(cycle)}

    # Per-block target from the marks: marked neighbour -> that key's centre.
    mark = mark_color(patterns)
    target = {nb: center for nb, center in key_marked_neighbours(patterns, uniform, mark)}

    def build(default):
        clicks = []
        for idx in sorted(uniform):
            cur = block_center_color(f, slots, idx, s)
            tgt = target.get(idx, default)
            n = (pos.get(tgt, pos[cur]) - pos[cur]) % ncy
            clicks += [block_click_xy(slots, idx, s)] * n
        return clicks

    for default in cycle:
        clicks = build(default)
        if plan_reaches_next_level(env, clicks, _click_xy):
            return clicks
    return None


def solve_multistate_key_board(env):
    """Decode a multi-state pattern-key board (cells cycle through k colours) and
    commit the completing block clicks. Thin composition of solve_click_board over
    the multi-state pattern-key planner."""
    return solve_click_board(env, discover_multistate_key_clicks)


def _gf2_solution(matrix, rhs, nvars):
    """Solve a binary linear system, choosing zero for every free variable."""
    a = [list(row) + [int(value)] for row, value in zip(matrix, rhs)]
    pivots = []
    row = 0
    for col in range(nvars):
        pivot = next((r for r in range(row, len(a)) if a[r][col]), None)
        if pivot is None:
            continue
        a[row], a[pivot] = a[pivot], a[row]
        for r in range(len(a)):
            if r != row and a[r][col]:
                a[r] = [x ^ y for x, y in zip(a[r], a[row])]
        pivots.append(col)
        row += 1
    if any(not any(r[:-1]) and r[-1] for r in a):
        return None
    out = [0] * nvars
    for r, col in enumerate(pivots):
        out[col] = a[r][-1]
    return out


def discover_coupled_key_clicks(env):
    """Plan a binary pattern-key board whose clicks may toggle several blocks.

    Some decorated blocks are inert clues. Their two neighbour symbols mean
    SAME/OPPOSITE relative to the clue's binary centre colour. Other decorated
    blocks are controls whose click effects are discovered on clones, just like
    the ordinary one-block controls. The decoded target constraints and observed
    click-effect matrix form a GF(2) system. Both possible SAME symbols are tried
    on clones; the reward identifies the correct interpretation. Unclued cells
    remain unconstrained. Returns the first reward-reaching click prefix.
    """
    # Coupled boards may decorate the controls themselves (for example with an
    # arrow showing the extra affected neighbour), so unlike the simpler key
    # variants they need not contain any fully uniform blocks.
    grid = _grid_of_blocks(env.frame())
    if not grid:
        return None
    s, _, slots, _, patterns = grid
    if not patterns:
        return None
    frame = np.asarray(env.frame())
    active = []
    effects = {}
    state_colours = set()

    # Discover which slots are controls and the parity effect of each click.
    for idx in sorted(slots):
        clone = env.clone()
        x, y = block_click_xy(slots, idx, s)
        clone.step(6, x, y)
        changed = {
            other for other in slots
            if block_center_color(frame, slots, other, s)
            != block_center_color(clone.frame(), slots, other, s)
        }
        if not changed:
            continue
        active.append(idx)
        effects[idx] = changed
        for other in changed:
            state_colours.add(block_center_color(frame, slots, other, s))
            state_colours.add(block_center_color(clone.frame(), slots, other, s))

    inert_clues = sorted(set(patterns) - set(active))
    states = sorted(state_colours)
    if not active or not inert_clues or len(states) != 2:
        return None

    # Each active neighbour may be described by several overlapping clues.
    observations = {}
    for idx in inert_clues:
        gg = read_mini_key(patterns[idx])
        center = block_center_color(frame, slots, idx, s)
        if center not in states:
            continue
        for di in range(-1, 2):
            for dj in range(-1, 2):
                if (di, dj) == (0, 0):
                    continue
                other = (idx[0] + di, idx[1] + dj)
                if other in effects:
                    observations.setdefault(other, []).append(
                        (gg[di + 1][dj + 1], center)
                    )
    tokens = sorted({token for obs in observations.values() for token, _ in obs})
    if len(tokens) != 2:
        return None

    base_level = env.levels_completed
    for same_token in tokens:
        targets = {}
        consistent = True
        for cell, obs in observations.items():
            wanted = set()
            for token, center in obs:
                opposite = states[1] if center == states[0] else states[0]
                wanted.add(center if token == same_token else opposite)
            if len(wanted) != 1:
                consistent = False
                break
            targets[cell] = next(iter(wanted))
        if not consistent or not targets:
            continue

        cells = sorted(targets)
        matrix = [
            [int(cell in effects[action]) for action in active]
            for cell in cells
        ]
        rhs = [
            int(block_center_color(frame, slots, cell, s) != targets[cell])
            for cell in cells
        ]
        solution = _gf2_solution(matrix, rhs, len(active))
        if solution is None:
            continue

        clicks = [
            block_click_xy(slots, action, s)
            for action, take in zip(active, solution) if take
        ]
        clone = env.clone()
        prefix = []
        for click in clicks:
            x, y = click
            clone.step(6, x, y)
            prefix.append(click)
            if clone.levels_completed > base_level:
                return prefix
    return None


def solve_coupled_key_board(env):
    """Decode and solve a binary key board with coupled toggle controls."""
    return solve_click_board(env, discover_coupled_key_clicks)

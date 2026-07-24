# Shared leg library: small, named, reusable skills.
# Players import from here; add a NEW leg only when no existing leg fits.
#
# r11l mechanics (discovered by probing raw frames):
#   Sole action is the coordinate click:  env.step(6, x, y)  with x=col, y=row.
#   There are TWO draggable ENDPOINTS and a BOX that is always their MIDPOINT.
#     - the ACTIVE endpoint is drawn as a small 15-pixel in a diamond of 0s ("cursor")
#     - the OTHER endpoint is a 15-pixel surrounded by 3s ("threeD")
#     - the BOX is a filled 5x5 block of 15 with a 6 in its centre.
#   A click moves the ACTIVE endpoint to (row=y, col=x).
#   Clicking ON the other endpoint's marker SELECTS it (it becomes active).
#   There is a hollow diamond of 15 (the RING / target); background is 5.
#   WIN: make the BOX (midpoint of the two endpoints) land exactly on the RING centre.
import numpy as np


def _arr(env):
    return np.asarray(env.frame())


# --- shared pixel-geometry helpers (each written ONCE) --------------------

def _neigh4_colors(f, r, c):
    """Set of colours in the 4-connected neighbourhood of (r,c) (in bounds)."""
    s = set()
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        rr, cc = r + dr, c + dc
        if 0 <= rr < f.shape[0] and 0 <= cc < f.shape[1]:
            s.add(int(f[rr, cc]))
    return s


def _neigh8_colors(f, r, c):
    """Set of colours in the 8-connected neighbourhood of (r,c) (in bounds)."""
    s = set()
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr, cc = r + dr, c + dc
            if 0 <= rr < f.shape[0] and 0 <= cc < f.shape[1]:
                s.add(int(f[rr, cc]))
    return s


def _centroid(points):
    """Rounded (row,col) centroid of a non-empty list of (row,col) points."""
    return (int(round(np.mean([p[0] for p in points]))),
            int(round(np.mean([p[1] for p in points]))))


def _isolated_cells(f, color):
    """Single-pixel (4-isolated) cells of `color` — endpoint/ring markers."""
    out = []
    for r, c in np.argwhere(f == color):
        iso = True
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < f.shape[0] and 0 <= cc < f.shape[1] and int(f[rr, cc]) == color:
                iso = False
                break
        if iso:
            out.append((int(r), int(c)))
    return out


def detect_rope(env):
    """Return (active_endpoint, other_endpoint, ring_center) as (row,col) tuples.

    active_endpoint  : the 15-pixel enclosed by 0s (the movable cursor)
    other_endpoint   : the 15-pixel enclosed by 3s
    ring_center      : centroid of the hollow diamond of 15 whose cells sit on
                       background (5) only.
    """
    f = _arr(env)
    active = other = None
    ringpts = []
    for (r, c) in np.argwhere(f == 15):
        nc = _neigh4_colors(f, r, c)
        if 0 in nc:
            active = (int(r), int(c))
        elif 3 in nc:
            other = (int(r), int(c))
        elif nc <= {5}:
            ringpts.append((int(r), int(c)))
    ring = _centroid(ringpts) if ringpts else None
    return active, other, ring


def click(env, row, col):
    """PRIMITIVE skill (written once): click the grid cell (row, col).

    In r11l this is the only action; its meaning depends on the target cell:
      - on empty space it moves the ACTIVE endpoint there;
      - on the other endpoint's marker it SELECTS that endpoint.
    Note the axis swap: the engine wants (x=col, y=row).
    """
    env.step(6, int(col), int(row))


def move_active_to(env, row, col):
    """Skill: move the currently active endpoint to (row, col)."""
    click(env, row, col)


def select_endpoint(env, point):
    """Skill: activate an endpoint by clicking its marker at (row, col)."""
    click(env, point[0], point[1])


def drag_endpoint(env, frm, dst):
    """Skill (written ONCE): pick up the endpoint whose marker is at `frm` and
    drop it at `dst`.  This is the recurring two-click gesture shared by every
    r11l level: SELECT the endpoint, then MOVE the now-active endpoint.

    Composition of the two primitive legs above, so both the single-rope
    (level 1) and multi-rope walk/straddle (level 2) solvers call the same
    gesture instead of re-issuing raw clicks.
    """
    select_endpoint(env, frm)
    move_active_to(env, dst[0], dst[1])


def place_box_on_ring(env):
    """Position the two rope endpoints symmetrically about the ring centre so the
    box (their midpoint) lands exactly on the ring, completing the level.

    Strategy: put both endpoints on the ring's row, at columns Rc-d and Rc+d,
    giving midpoint = ring centre. Choose d so both columns stay on-grid.
    """
    active, other, ring = detect_rope(env)
    if ring is None or active is None or other is None:
        return
    Rr, Rc = ring
    W = _arr(env).shape[1]
    d = min(9, Rc, (W - 1) - Rc)
    # 1) move the already-active endpoint onto (Rr, Rc-d)
    move_active_to(env, Rr, Rc - d)
    # 2) drag the OTHER endpoint onto (Rr, Rc+d); midpoint == ring centre -> win
    drag_endpoint(env, other, (Rr, Rc + d))


# ---------------------------------------------------------------------------
# r11l level 2+ : several colour-coded rope systems sharing one cursor.
#   Each colour C that owns BOTH a filled box (a 6 pixel wrapped in colour C)
#   and a hollow diamond RING (single C pixels touching only background 5) is
#   a system.  The box always sits at the (rounded) CENTROID of that colour's
#   endpoints.  WIN = every box centred on its own-colour ring.
#
#   Mechanics recap (verified on clones): the cursor is pick/drop.  Clicking an
#   endpoint marker picks it up (it becomes the 0-diamond ACTIVE); the next
#   click drops it.  Each drop is bounded by a rope length from the box, so a
#   distant box must be WALKED (move one endpoint toward the target, the box
#   drifts, repeat).  Nearby targets are reachable in one straddle.
# ---------------------------------------------------------------------------

def box_center(env, color):
    """(row,col) of the box (the 6 pixel wrapped in `color`) for a colour."""
    f = _arr(env)
    for r, c in np.argwhere(f == 6):
        blk = f[max(0, r - 2):r + 3, max(0, c - 2):c + 3]
        if color in blk:
            return (int(r), int(c))
    return None


def active_pos(env):
    """(row,col) of the currently picked-up (ACTIVE, 0-diamond) endpoint."""
    f = _arr(env)
    for color in np.unique(f):
        if color in (0, 1, 2, 3, 5, 6):
            continue
        for b in _isolated_cells(f, int(color)):
            r, c = b
            if 0 in _neigh8_colors(f, r, c):
                return (r, c)
    return None


def endpoints_of(env, color):
    """List of (row,col) endpoint markers of `color` (0- or 3-diamond centres)."""
    f = _arr(env)
    out = []
    for (r, c) in _isolated_cells(f, color):
        nb = _neigh8_colors(f, r, c)
        if 0 in nb or 3 in nb:
            out.append((r, c))
    return sorted(out)


def ring_center(env, color):
    """(row,col) centre of the hollow diamond RING drawn in `color`."""
    f = _arr(env)
    ring = [(r, c) for (r, c) in _isolated_cells(f, color)
            if _neigh4_colors(f, r, c) <= {5}]
    return _centroid(ring) if ring else None


def ring_systems(env):
    """Discover every colour owning both a box and a ring.

    Returns list of dicts: {color, ring, box, endpoints}.
    """
    f = _arr(env)
    systems = []
    for color in np.unique(f):
        color = int(color)
        if color in (0, 1, 2, 3, 5, 6):
            continue
        ring = ring_center(env, color)
        box = box_center(env, color)
        if ring is None or box is None:
            continue
        systems.append({
            "color": color,
            "ring": ring,
            "box": box,
            "endpoints": endpoints_of(env, color),
        })
    return systems


def probe_drag(env, frm, dst, base_levels=None):
    """Skill (written ONCE): TRY one `drag_endpoint` on a CLONE and report the
    outcome without mutating `env`.

    This is the recurring "look before you leap" gesture shared by every greedy
    r11l solver: pick the endpoint at `frm`, drop it at `dst`, then inspect the
    resulting clone.  Returns (clone, landed, completed):
      clone     : the env AFTER the trial drag (inspect its frame freely);
      landed    : True iff the ACTIVE endpoint actually reached `dst`
                  (a rope-length-bounded drop that overshoots does not land);
      completed : True iff the drag increased levels_completed (this move wins).
    `base_levels` defaults to the caller's current level count.
    """
    if base_levels is None:
        base_levels = env.levels_completed
    c = env.clone()
    drag_endpoint(c, frm, dst)
    completed = c.levels_completed > base_levels
    landed = active_pos(c) == (dst[0], dst[1])
    return c, landed, completed


def _can_drop(env, frm, dst):
    """True iff picking the endpoint at `frm` and dropping at `dst` succeeds."""
    _, landed, _ = probe_drag(env, frm, dst)
    return landed


def walk_box_to(env, color, target, eps, iters=120):
    """Rope-walk a box's centroid onto `target` by greedily moving whichever
    endpoint (to a reachable cell) brings the box nearest the target.

    `eps` is a mutable list of that colour's endpoint positions (bookkept as we
    move them, since markers can hide under the box/ring).  Commits on `env`.
    Returns the final box position.
    """
    start_levels = env.levels_completed
    for _ in range(iters):
        if env.terminal() or env.levels_completed > start_levels:
            return None
        b = box_center(env, color)
        if b is None or b == target:
            return b
        best = None
        for i, e in enumerate(eps):
            dsts = set()
            for frac in (1.0, .9, .8, .7, .6, .5, .4, .3, .2, .12, .07, .04):
                dsts.add((int(round(e[0] + (target[0] - e[0]) * frac)),
                          int(round(e[1] + (target[1] - e[1]) * frac))))
            for dr in range(-8, 9, 2):
                for dc in range(-8, 9, 2):
                    dsts.add((target[0] + dr, target[1] + dc))
            for dst in dsts:
                if not (0 <= dst[0] < 64 and 0 <= dst[1] < 64) or dst == e:
                    continue
                if any(abs(dst[0] - o[0]) < 3 and abs(dst[1] - o[1]) < 3
                       for j, o in enumerate(eps) if j != i):
                    continue
                c, landed, completed = probe_drag(env, e, dst, start_levels)
                if completed:
                    best = (-1, i, e, dst)  # this move completes the level
                    break
                if not landed:
                    continue
                nb = box_center(c, color)
                if nb is None:
                    continue
                d = abs(nb[0] - target[0]) + abs(nb[1] - target[1])
                if best is None or d < best[0]:
                    best = (d, i, e, dst)
            if best is not None and best[0] == -1:
                break
        if best is None:
            break
        _, i, e, dst = best
        drag_endpoint(env, e, dst)
        eps[i] = dst
        if env.levels_completed > start_levels or env.terminal():
            return None
    return box_center(env, color)


def _run_straddle_plan(target_env, eps, perm, pat, target, base_levels,
                       check_bounds):
    """Skill (written ONCE): apply ONE symmetric-straddle plan on `target_env`.

    A "plan" is a permutation `perm` assigning endpoints to the offset pattern
    `pat`; endpoint `perm[slot]` is dragged to `target + pat[slot]`.  Drags run
    in slot order and stop early the instant the level completes.  `check_bounds`
    aborts the plan if an offset would leave the 64x64 grid (used when TRIALLING
    on a clone; skipped when COMMITTING a plan the trial already validated).

    Returns the updated endpoint-position list `cur`.  Because `straddle_box_to`
    used to inline this exact loop twice (once to test on a clone, once to
    commit on the real env), both now call this single leg.
    """
    cur = [tuple(x) for x in eps]
    for slot, i in enumerate(perm):
        dst = (target[0] + pat[slot][0], target[1] + pat[slot][1])
        if check_bounds and not (0 <= dst[0] < 64 and 0 <= dst[1] < 64):
            break
        if cur[i] == dst:
            continue
        drag_endpoint(target_env, cur[i], dst)
        cur[i] = dst
        if target_env.levels_completed > base_levels:
            break
    return cur


def straddle_box_to(env, color, target, eps):
    """Place a box exactly on `target` by scattering its endpoints symmetrically
    about the target (offsets summing to zero -> centroid == target).

    Tries several offset magnitudes / orientations and endpoint assignments;
    each candidate is verified on a clone before committing.  Returns True on
    success.  Works when the target is within one rope reach of every endpoint.
    """
    from itertools import permutations
    if box_center(env, color) == target:
        return True
    n = len(eps)
    patterns = []
    if n == 2:
        for a in range(3, 12):
            patterns += [[(-a, a), (a, -a)], [(-a, -a), (a, a)],
                         [(-a, 0), (a, 0)], [(0, -a), (0, a)],
                         [(-a, 1), (a, -1)], [(-1, a), (1, -a)]]
    elif n == 3:
        for k in range(3, 9):
            patterns += [[(-2 * k, 0), (k, -k), (k, k)],
                         [(2 * k, 0), (-k, -k), (-k, k)],
                         [(0, -2 * k), (-k, k), (k, k)],
                         [(0, 2 * k), (-k, -k), (k, -k)],
                         [(-k, -k), (k, k), (0, 0)],
                         [(-k, 0), (k, -k), (0, k)]]
    start_levels = env.levels_completed
    for pat in patterns:
        for perm in set(permutations(range(n))):
            # TRIAL the plan on a clone (bounds-checked).
            c = env.clone()
            _run_straddle_plan(c, eps, perm, pat, target, start_levels,
                               check_bounds=True)
            # Success if the box snapped onto the target, OR the final move
            # already completed the level (which mutates the frame away).
            if c.levels_completed > start_levels or box_center(c, color) == target:
                # COMMIT the (already-validated) plan on the real env.
                cur = _run_straddle_plan(env, eps, perm, pat, target,
                                         start_levels, check_bounds=False)
                for i in range(n):
                    if i < len(cur):
                        eps[i] = cur[i]
                return True
    return False


# ---------------------------------------------------------------------------
# r11l level 4+ : same rope/box/ring mechanic, but boxes AND rings are
# MULTI-COLOURED.  A box is a solid diamond whose coloured parts (e.g. a
# 12/14 top and a 15 bottom) must be dropped into the matching hollow-diamond
# RING that shows the SAME colour parts.  Decoy rings (colour sets that match
# no box) litter the board.  The identity of a rope (which box its endpoints
# drive) is discovered by PROBING, not by colour, so this stays general.
# ---------------------------------------------------------------------------

def _flood_nonbg(f, r, c, bg=5):
    """Flood the 4-connected non-`bg` region from (r,c); return (colors,size)."""
    from collections import deque
    seen = set()
    q = deque([(r, c)])
    cols = set()
    while q:
        y, x = q.popleft()
        if (y, x) in seen:
            continue
        seen.add((y, x))
        if int(f[y, x]) == bg:
            continue
        cols.add(int(f[y, x]))
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < f.shape[0] and 0 <= nx < f.shape[1] \
               and (ny, nx) not in seen and int(f[ny, nx]) != bg:
                q.append((ny, nx))
    return cols, len(seen)


def multicolor_boxes(env, min_size=8):
    """Boxes on a multi-coloured board: each is a solid blob of >= `min_size`
    cells around a `6` centre.  Returns list of {pos, colors} where `colors`
    is the fill palette (excluding rope/marker colours 0,1,3 and 5,6).
    """
    f = _arr(env)
    out = []
    for r, c in np.argwhere(f == 6):
        cols, n = _flood_nonbg(f, int(r), int(c))
        if n >= min_size:
            out.append({"pos": (int(r), int(c)),
                        "colors": frozenset(cols - {0, 1, 3, 5, 6})})
    return out


def _ring_outline_cells(f):
    """Isolated single-colour cells (4-neighbourhood is only background) that
    form hollow-diamond ring outlines; excludes rope/marker colours 0,1,3,6."""
    out = []
    rows, cols = f.shape[:2]
    for r, c in np.argwhere(f != 5):
        col = int(f[r, c])
        if col in (0, 1, 3):
            continue
        ok = True
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < rows and 0 <= cc < cols and int(f[rr, cc]) != 5:
                ok = False
                break
        if ok:
            out.append((int(r), int(c), col))
    return out


def multicolor_rings(env, radius=5):
    """Cluster ring-outline cells into hollow diamonds.  Returns list of
    {center, colors} where colours is the palette of that ring cluster."""
    cells = _ring_outline_cells(_arr(env))
    clusters = []
    for (r, c, col) in cells:
        placed = False
        for cl in clusters:
            if any(abs(r - rr) <= radius and abs(c - cc) <= radius
                   for rr, cc, _ in cl):
                cl.append((r, c, col))
                placed = True
                break
        if not placed:
            clusters.append([(r, c, col)])
    merged = True
    while merged:
        merged = False
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                a, b = clusters[i], clusters[j]
                if any(abs(r1 - r2) <= radius and abs(c1 - c2) <= radius
                       for r1, c1, _ in a for r2, c2, _ in b):
                    clusters[i] = a + b
                    clusters.pop(j)
                    merged = True
                    break
            if merged:
                break
    out = []
    for cl in clusters:
        rs = [r for r, c, _ in cl]
        cs = [c for r, c, _ in cl]
        out.append({"center": (int(round(np.mean(rs))), int(round(np.mean(cs)))),
                    "colors": frozenset(x for _, _, x in cl)})
    return out


def marker_endpoints(env):
    """Precise rope endpoint markers: single coloured pixels whose four
    orthogonal neighbours are ALL the diamond-wrap colour 0 (active cursor) or
    ALL 3 (idle endpoint).  Filters out border artefacts."""
    f = _arr(env)
    rows, cols = f.shape[:2]
    out = []
    for r, c in np.argwhere(~np.isin(f, [0, 1, 3, 5, 6])):
        r, c = int(r), int(c)
        nb = []
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < rows and 0 <= cc < cols:
                nb.append(int(f[rr, cc]))
        if len(nb) == 4 and (all(v == 0 for v in nb) or all(v == 3 for v in nb)):
            out.append((r, c))
    return out


def _box_centers_set(env, min_size=8):
    return {b["pos"] for b in multicolor_boxes(env, min_size)}


def group_endpoints_by_box(env):
    """PROBE-discover which box each endpoint marker drives (general, no colour
    assumptions).  For every marker: on a clone select it and nudge it, then see
    which box's 6-centre moved.  Returns {box_pos: [endpoint markers...]}."""
    eps = marker_endpoints(env)
    groups = {}
    for ep in eps:
        c = env.clone()
        click(c, ep[0], ep[1])              # select (no-op if already active)
        ap = active_pos(c)
        if ap is None:
            continue
        before = _box_centers_set(c)
        click(c, ap[0], min(ap[1] + 5, 63))  # nudge
        after = _box_centers_set(c)
        moved = before - after
        if len(moved) == 1:
            groups.setdefault(next(iter(moved)), []).append(ep)
    return groups


def multicolor_systems(env):
    """Full level-4 system list: match each multi-coloured box to the ring whose
    colour palette is identical, discover its endpoints by probing, and choose a
    tracking colour unique to that box (so `box_center`/`ring` stay unambiguous).

    Returns list of {color, box, ring, endpoints}, where `color` is that unique
    tracking colour.  This is the SAME shape as `ring_systems`, so both feed the
    shared `solve_systems_walk_straddle` leg unchanged.
    """
    boxes = multicolor_boxes(env)
    rings = multicolor_rings(env)
    groups = group_endpoints_by_box(env)
    # colours appearing in more than one box (ambiguous for tracking)
    from collections import Counter
    cnt = Counter()
    for b in boxes:
        for col in b["colors"]:
            cnt[col] += 1
    systems = []
    for b in boxes:
        match = [r for r in rings if r["colors"] == b["colors"]]
        if not match:
            # fall back to best overlap
            match = sorted(rings, key=lambda r: -len(r["colors"] & b["colors"]))
            if not match or not (match[0]["colors"] & b["colors"]):
                continue
        ring = min(match, key=lambda r: abs(r["center"][0] - b["pos"][0]) +
                   abs(r["center"][1] - b["pos"][1]))
        uniq = min((col for col in b["colors"] if cnt[col] == 1),
                   default=min(b["colors"]))
        systems.append({"color": int(uniq),
                        "box": b["pos"], "ring": ring["center"],
                        "endpoints": sorted(groups.get(b["pos"], []))})
    return systems


def solve_systems_walk_straddle(env, systems):
    """HIGHER-ORDER composition leg (written ONCE): solve a whole board of
    independent rope systems by walk-then-straddle.

    A `systems` entry is a dict with the uniform shape produced by BOTH
    `ring_systems` (levels 2/3) and `multicolor_systems` (level 4):
      color     : a tracking colour UNIQUE to that box (so `box_center`/rings
                  stay unambiguous);
      box       : the box's current (row,col) centre (used only for ordering);
      ring      : the (row,col) ring centre this box must land on;
      endpoints : that rope's endpoint markers.

    This captures the recurring pattern shared by every multi-rope solver:
      1. order the systems FARTHEST box-to-ring gap first, so distant boxes are
         rope-walked in before nearby ones are exactly straddle-snapped;
      2. for each system, skip if already solved, else rope-WALK the box close
         to its ring (which also clusters its endpoints out of other systems'
         way) and then STRADDLE-snap it exactly onto the ring centre;
      3. stop the instant a drop completes the level / terminates.

    Nothing here assumes HOW the systems were discovered, so any discovery leg
    that yields the uniform shape above can be plugged straight in.
    """
    start_levels = env.levels_completed
    systems = sorted(systems, key=lambda s: -(abs(s["box"][0] - s["ring"][0]) +
                                              abs(s["box"][1] - s["ring"][1])))
    for s in systems:
        if env.terminal() or env.levels_completed > start_levels:
            break
        color, ring = s["color"], s["ring"]
        if box_center(env, color) == ring:
            continue
        eps = list(s["endpoints"])
        walk_box_to(env, color, ring, eps)
        if env.terminal() or env.levels_completed > start_levels:
            break
        straddle_box_to(env, color, ring, eps)


def place_multicolor_boxes(env):
    """Solve a level-4-style board: drop every multi-coloured box into its
    colour-matching hollow-diamond ring (decoy rings are skipped by the
    discovery leg).  Thin composition: discover the systems, then hand them to
    the shared walk-then-straddle solver."""
    solve_systems_walk_straddle(env, multicolor_systems(env))


def place_boxes_on_rings(env):
    """Solve a multi-rope level: centre every colour's box on its own ring.
    Thin composition: discover the colour systems, then hand them to the shared
    walk-then-straddle solver."""
    solve_systems_walk_straddle(env, ring_systems(env))


def _components_matching(f, wanted):
    """Return 4-connected components selected by a boolean cell mask."""
    seen = np.zeros(f.shape, dtype=bool)
    out = []
    for r, c in np.argwhere(wanted):
        r, c = int(r), int(c)
        if seen[r, c]:
            continue
        seen[r, c] = True
        stack, cells = [(r, c)], []
        while stack:
            y, x = stack.pop()
            cells.append((y, x))
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                yy, xx = y + dy, x + dx
                if (0 <= yy < f.shape[0] and 0 <= xx < f.shape[1]
                        and wanted[yy, xx] and not seen[yy, xx]):
                    seen[yy, xx] = True
                    stack.append((yy, xx))
        out.append(cells)
    return out


def _color_components(f, color):
    """Return the 4-connected components of one colour as cell lists."""
    return _components_matching(f, f == color)


def _nonbackground_components(f, background=5):
    """Return components formed by any mutually touching non-background cells."""
    return _components_matching(f, f != background)


def assembly_pieces(env):
    """Discover the one-palette solid diamonds that can be picked up.

    An uncollected piece is a compact 5x5 diamond split between empty black
    slots and one real palette.  Hollow ring pixels are disconnected, while
    rope carriers are connected to ropes/endpoints, so this geometric test
    separates the affordances without level coordinates.
    """
    f = _arr(env)
    out = []
    for cells in _nonbackground_components(f):
        rs, cs = zip(*cells)
        bbox = (min(rs), min(cs), max(rs), max(cs))
        colors = frozenset(int(f[r, c]) for r, c in cells)
        if (18 <= len(cells) <= 25 and bbox[2] - bbox[0] == 4
                and bbox[3] - bbox[1] == 4 and 0 in colors
                and len(colors) == 2):
            out.append({"pos": ((bbox[0] + bbox[2]) // 2,
                                (bbox[1] + bbox[3]) // 2),
                        "colors": colors - {0}})
    return sorted(out, key=lambda x: x["pos"])


def centroid_carrier_systems(env):
    """Discover black carriers and the marker endpoints driving each carrier.

    The carrier is the rounded centroid of its endpoint group.  We solve the
    tiny exact partition of observed markers rather than assuming every rope
    has two ends; level 5 deliberately mixes a two-end and a three-end carrier.
    """
    from itertools import combinations
    f = _arr(env)
    eps = marker_endpoints(env)
    carriers = []
    for cells in _color_components(f, 0):
        rs, cs = zip(*cells)
        h, w = max(rs) - min(rs) + 1, max(cs) - min(cs) + 1
        if 15 <= len(cells) <= 25 and h <= 5 and w <= 5:
            carriers.append(_centroid(cells))
    carriers = sorted(set(carriers))
    candidates = {}
    for center in carriers:
        candidates[center] = []
        for n in range(2, len(eps) + 1):
            for subset in combinations(eps, n):
                if _centroid(subset) == center:
                    candidates[center].append(subset)

    def partition(i, used, chosen):
        if i == len(carriers):
            return chosen if len(used) == len(eps) else None
        center = carriers[i]
        for subset in candidates[center]:
            if not (set(subset) & used):
                found = partition(i + 1, used | set(subset),
                                  chosen + [{"box": center,
                                             "endpoints": list(subset)}])
                if found is not None:
                    return found
        return None

    return partition(0, set(), []) or []


def assembly_targets(env, pieces):
    """Choose a disjoint exact cover of piece palettes by ring palettes.

    Rings requiring unavailable colours are decoys.  A tempting singleton
    ring is also rejected when using it would leave another piece uncovered.
    """
    rings = multicolor_rings(env, radius=3)
    candidates = []
    for ring_i, ring in enumerate(rings):
        for mask in range(1, 1 << len(pieces)):
            palette, indices = set(), []
            for i, piece in enumerate(pieces):
                if mask & (1 << i):
                    palette.update(piece["colors"])
                    indices.append(i)
            if frozenset(palette) == ring["colors"]:
                candidates.append({"ring_i": ring_i, "ring": ring["center"],
                                   "pieces": indices})

    all_used = set(range(len(pieces)))

    def cover(used, used_rings, plans):
        if used == all_used:
            return plans
        first = min(all_used - used)
        for candidate in candidates:
            chosen = set(candidate["pieces"])
            if (first in chosen and not (chosen & used)
                    and candidate["ring_i"] not in used_rings):
                found = cover(used | chosen,
                              used_rings | {candidate["ring_i"]},
                              plans + [candidate])
                if found is not None:
                    return found
        return None

    return cover(set(), set(), []) or []


def _carrier_patterns(n):
    """Zero-sum endpoint offsets whose centroid is exactly the target."""
    patterns = []
    if n == 2:
        for a in range(4, 12):
            patterns += [[(0, -a), (0, a)], [(-a, 0), (a, 0)],
                         [(-a, -a), (a, a)], [(-a, a), (a, -a)]]
    elif n == 3:
        for k in range(3, 9):
            patterns += [[(-2 * k, 0), (k, -k), (k, k)],
                         [(2 * k, 0), (-k, -k), (-k, k)],
                         [(0, -2 * k), (-k, k), (k, k)],
                         [(0, 2 * k), (-k, -k), (k, -k)]]
    return patterns


def _apply_carrier_pattern(env, endpoints, destinations, base_level):
    """Apply one endpoint pattern, updating its marker positions in place."""
    for i, destination in enumerate(destinations):
        drag_endpoint(env, endpoints[i], destination)
        endpoints[i] = destination
        if env.levels_completed > base_level:
            break


def move_carrier_to(env, endpoints, target):
    """Move one centroid carrier onto a pickup or ring, clone-verifying space.

    This is the dense-progress primitive: after every call the carrier centroid
    is observably at one planned pickup/target, regardless of sparse reward.
    The mutable `endpoints` list is updated to the marker positions used.
    """
    base_level = int(env.levels_completed)
    for pattern in _carrier_patterns(len(endpoints)):
        destinations = [(target[0] + dr, target[1] + dc)
                        for dr, dc in pattern]
        if any(not (0 <= r < 64 and 0 <= c < 64)
               for r, c in destinations):
            continue
        trial = env.clone()
        trial_eps = list(endpoints)
        _apply_carrier_pattern(trial, trial_eps, destinations, base_level)
        if (trial.levels_completed <= base_level
                and not set(destinations) <= set(marker_endpoints(trial))):
            continue
        _apply_carrier_pattern(env, endpoints, destinations, base_level)
        return True
    return False


def _execute_assembly(env, systems, plans, pieces):
    """Collect every plan's pieces into one carrier, then deliver the union."""
    for system, plan in zip(systems, plans):
        endpoints = system["endpoints"]
        for piece_i in plan["pieces"]:
            if not move_carrier_to(env, endpoints, pieces[piece_i]["pos"]):
                return False
        if not move_carrier_to(env, endpoints, plan["ring"]):
            return False
    return True


def assemble_pieces_on_rings(env):
    """Solve a carrier-assembly board with palette-exact rings and decoys.

    Each carrier gathers all solid piece palettes required by one chosen ring.
    Carrier-to-plan reachability/order is verified end-to-end on clones before
    the validated assignment is committed to the real environment.
    """
    from itertools import permutations
    pieces = assembly_pieces(env)
    systems = centroid_carrier_systems(env)
    plans = assembly_targets(env, pieces)
    if not pieces or not plans or len(plans) > len(systems):
        return False
    base_level = int(env.levels_completed)
    for assignment in permutations(systems, len(plans)):
        trial = env.clone()
        trial_systems = [{"box": s["box"],
                          "endpoints": list(s["endpoints"])}
                         for s in assignment]
        if (_execute_assembly(trial, trial_systems, plans, pieces)
                and trial.levels_completed > base_level):
            real_systems = [{"box": s["box"],
                             "endpoints": list(s["endpoints"])}
                            for s in assignment]
            _execute_assembly(env, real_systems, plans, pieces)
            return env.levels_completed > base_level
    return False


# --- partial / barrier-board assembly -------------------------------------

def _cluster_endpoint_groups(endpoints, count):
    """Agglomeratively split spatial marker endpoints into `count` carriers.

    Some boards draw a carrier touching its ropes, so black-component carrier
    detection is intentionally unavailable there.  Endpoints belonging to one
    carrier still form the tight spatial cluster around its centroid.
    """
    groups = [[point] for point in sorted(endpoints)]
    while len(groups) > count:
        best = None
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                distance = min(abs(a[0] - b[0]) + abs(a[1] - b[1])
                               for a in groups[i] for b in groups[j])
                candidate = (distance, i, j)
                if best is None or candidate < best:
                    best = candidate
        _, i, j = best
        groups[i] = sorted(groups[i] + groups[j])
        groups.pop(j)
    return groups


def partial_assembly_targets(env, pieces, limit):
    """Find the strongest disjoint set of rings exactly buildable by pieces.

    Unlike :func:`assembly_targets`, unmatched pieces are allowed: barrier
    boards contain deliberately unreachable decoys.  Exact palette equality
    still distinguishes real goals from rings requiring unavailable colours.
    """
    rings = multicolor_rings(env, radius=3)
    by_ring = []
    for ring in rings:
        candidates = []
        for mask in range(1, 1 << len(pieces)):
            indices = [i for i in range(len(pieces)) if mask & (1 << i)]
            palette = frozenset().union(
                *(pieces[i]["colors"] for i in indices))
            if palette == ring["colors"]:
                candidates.append({"ring": ring["center"],
                                   "pieces": indices})
        by_ring.append(candidates)

    best = []

    def choose(ring_i, used, plans):
        nonlocal best
        score = (sum(len(p["pieces"]) for p in plans), len(plans))
        best_score = (sum(len(p["pieces"]) for p in best), len(best))
        if score > best_score:
            best = list(plans)
        if ring_i == len(by_ring) or len(plans) == limit:
            return
        choose(ring_i + 1, used, plans)
        for candidate in by_ring[ring_i]:
            selected = set(candidate["pieces"])
            if not (selected & used):
                choose(ring_i + 1, used | selected,
                       plans + [candidate])

    choose(0, set(), [])
    return best


def assemble_reachable_pieces_on_rings(env):
    """Assemble every exactly buildable target while ignoring decoy pieces.

    This composes the established centroid movement/clone-verification leg.
    The only new discovery is grouping endpoints spatially when carrier blobs
    merge into ropes, and selecting a maximum disjoint partial exact cover.
    """
    from itertools import permutations
    pieces = assembly_pieces(env)
    endpoints = marker_endpoints(env)
    max_systems = len(endpoints) // 2
    plans = partial_assembly_targets(env, pieces, max_systems)
    if not pieces or not plans:
        return False
    groups = _cluster_endpoint_groups(endpoints, len(plans))
    systems = [{"box": _centroid(group), "endpoints": group}
               for group in groups]
    base_level = int(env.levels_completed)
    for assignment in permutations(systems):
        trial = env.clone()
        trial_systems = [{"box": s["box"],
                          "endpoints": list(s["endpoints"])}
                         for s in assignment]
        if (_execute_assembly(trial, trial_systems, plans, pieces)
                and trial.levels_completed > base_level):
            real_systems = [{"box": s["box"],
                             "endpoints": list(s["endpoints"])}
                            for s in assignment]
            _execute_assembly(env, real_systems, plans, pieces)
            return env.levels_completed > base_level
    return False

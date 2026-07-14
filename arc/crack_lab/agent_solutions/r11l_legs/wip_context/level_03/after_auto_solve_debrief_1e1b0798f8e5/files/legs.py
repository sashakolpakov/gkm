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


def place_boxes_on_rings(env):
    """Solve a multi-rope level: centre every colour's box on its own ring.

    Systems whose box is far from its ring (needs rope-walking) are handled
    first; nearby ones are straddled.  Straddle is tried first for each system
    (cheap, exact); if it cannot reach, the box is walked, then straddled to
    snap exactly onto the ring centre.
    """
    start_levels = env.levels_completed
    systems = ring_systems(env)
    # walk-first ordering: farthest box-to-ring gap first
    systems.sort(key=lambda s: -(abs(s["box"][0] - s["ring"][0]) +
                                 abs(s["box"][1] - s["ring"][1])))
    for s in systems:
        if env.terminal() or env.levels_completed > start_levels:
            break
        color, ring = s["color"], s["ring"]
        if box_center(env, color) == ring:
            continue
        eps = endpoints_of(env, color)
        # Always rope-walk first: this drags the endpoints in close to the ring
        # (clustered around it), which both makes an exact straddle-snap cheap
        # AND keeps this colour's endpoints out of the other systems' way.
        walk_box_to(env, color, ring, eps)
        if env.terminal() or env.levels_completed > start_levels:
            break
        straddle_box_to(env, color, ring, eps)

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


def _neigh_colors(f, r, c):
    s = set()
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        rr, cc = r + dr, c + dc
        if 0 <= rr < f.shape[0] and 0 <= cc < f.shape[1]:
            s.add(int(f[rr, cc]))
    return s


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
        nc = _neigh_colors(f, r, c)
        if 0 in nc:
            active = (int(r), int(c))
        elif 3 in nc:
            other = (int(r), int(c))
        elif nc <= {5}:
            ringpts.append((int(r), int(c)))
    ring = None
    if ringpts:
        ring = (int(round(np.mean([p[0] for p in ringpts]))),
                int(round(np.mean([p[1] for p in ringpts]))))
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
    # 1) move the active endpoint onto (Rr, Rc-d)
    move_active_to(env, Rr, Rc - d)
    # 2) select the other endpoint by clicking its marker
    select_endpoint(env, other)
    # 3) move it onto (Rr, Rc+d); midpoint == ring centre -> win
    move_active_to(env, Rr, Rc + d)


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

def _neigh8(f, r, c):
    s = set()
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr, cc = r + dr, c + dc
            if 0 <= rr < f.shape[0] and 0 <= cc < f.shape[1]:
                s.add(int(f[rr, cc]))
    return s


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
        for b in _components_area1(f, int(color)):
            r, c = b
            if 0 in _neigh8(f, r, c):
                return (r, c)
    return None


def _components_area1(f, color):
    """Return single-pixel (isolated) cells of `color` (endpoint/ring markers)."""
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


def endpoints_of(env, color):
    """List of (row,col) endpoint markers of `color` (0- or 3-diamond centres)."""
    f = _arr(env)
    out = []
    for (r, c) in _components_area1(f, color):
        nb = _neigh8(f, r, c)
        if 0 in nb or 3 in nb:
            out.append((r, c))
    return sorted(out)


def ring_center(env, color):
    """(row,col) centre of the hollow diamond RING drawn in `color`."""
    f = _arr(env)
    ring = []
    for (r, c) in _components_area1(f, color):
        nb = set()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < f.shape[0] and 0 <= cc < f.shape[1]:
                nb.add(int(f[rr, cc]))
        if nb <= {5}:
            ring.append((r, c))
    if not ring:
        return None
    return (int(round(np.mean([p[0] for p in ring]))),
            int(round(np.mean([p[1] for p in ring]))))


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


def _can_drop(env, frm, dst):
    """True iff picking the endpoint at `frm` and dropping at `dst` succeeds."""
    c = env.clone()
    click(c, frm[0], frm[1])
    click(c, dst[0], dst[1])
    return active_pos(c) == (dst[0], dst[1])


def _drop(env, frm, dst):
    click(env, frm[0], frm[1])
    click(env, dst[0], dst[1])


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
                c = env.clone()
                click(c, e[0], e[1])
                click(c, dst[0], dst[1])
                if c.levels_completed > start_levels:
                    best = (-1, i, e, dst)  # this move completes the level
                    break
                if active_pos(c) != (dst[0], dst[1]):
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
        _drop(env, e, dst)
        eps[i] = dst
        if env.levels_completed > start_levels or env.terminal():
            return None
    return box_center(env, color)


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
            c = env.clone()
            cur = [tuple(x) for x in eps]
            for slot, i in enumerate(perm):
                dst = (target[0] + pat[slot][0], target[1] + pat[slot][1])
                if not (0 <= dst[0] < 64 and 0 <= dst[1] < 64):
                    break
                if cur[i] == dst:
                    continue
                _drop(c, cur[i], dst)
                cur[i] = dst
                if c.levels_completed > start_levels:
                    break
            # Success if the box snapped onto the target, OR the final move
            # already completed the level (which mutates the frame away).
            if c.levels_completed > start_levels or box_center(c, color) == target:
                cur = [tuple(x) for x in eps]
                for slot, i in enumerate(perm):
                    dst = (target[0] + pat[slot][0], target[1] + pat[slot][1])
                    if cur[i] == dst:
                        continue
                    _drop(env, cur[i], dst)
                    cur[i] = dst
                    if env.levels_completed > start_levels:
                        break
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

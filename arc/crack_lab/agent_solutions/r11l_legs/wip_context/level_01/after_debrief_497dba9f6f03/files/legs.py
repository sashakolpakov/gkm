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

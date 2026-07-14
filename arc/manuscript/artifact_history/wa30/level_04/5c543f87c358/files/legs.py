# Shared leg library: small, named, reusable skills.
# Players import from here; add a NEW leg only when no existing leg fits.
#
# World model discovered by experiment on wa30:
#  - Everything lives on a 4-pixel logical grid (one "cell" = 4x4 pixels).
#  - The avatar is colour 14 body + colour 0 "nose"; it occupies a 4x4 footprint
#    and each movement action translates it exactly 4px. Actions:
#        1=up 2=down 3=left 4=right, 5=effect (grab / release).
#  - Movable boxes: 4x4, colour-4 border + colour-9 interior.
#  - Walking into a box does not push it usefully; instead, when the nose touches
#    a box you press 5 to GRAB it. While carried the box rides rigidly with the
#    avatar; press 5 again to RELEASE (dropping it, border recolours to 3).
#  - A target/container region: colour-9 border + colour-2 interior. The goal is
#    to fill every target cell with a box; reward (levels_completed) fires only
#    once ALL are filled.
import numpy as np
from collections import Counter, deque

VEC = {1: (-4, 0), 2: (4, 0), 3: (0, -4), 4: (0, 4)}  # action -> (drow, dcol)


def foot(frame):
    """Top-left (row, col) of the avatar's 4x4 footprint (colours 14 and 0)."""
    ys, xs = np.where((frame == 14) | (frame == 0))
    return int(ys.min()), int(xs.min())


def clusters(frame, color, min_cells=8):
    """Find 4x4-ish blobs of `color`; return list of (top,left). Filters out
    stray HUD pixels via min_cells."""
    ys, xs = np.where(frame == color)
    pts = set(zip(ys.tolist(), xs.tolist()))
    seen, res = set(), []
    for p in sorted(pts):
        if p in seen:
            continue
        cl = [q for q in pts if abs(q[0] - p[0]) < 4 and abs(q[1] - p[1]) < 4]
        for q in cl:
            seen.add(q)
        if len(cl) >= min_cells:
            res.append((min(q[0] for q in cl), min(q[1] for q in cl)))
    return res


def _align4(vals):
    """Unique 4-aligned starts ((v//4)*4) for a 1-D iterable of pixel coords,
    sorted ascending. Empty in -> empty out. The one place the 4px snap lives."""
    return sorted({(int(v) // 4) * 4 for v in vals})


def target_cells(frame, interior=2):
    """From the container's interior colour, compute the 4-aligned target cells
    (top,left) that tile it. Returns list of (row, col)."""
    ys, xs = np.where(frame == interior)
    if not len(ys):
        return []
    r0 = _align4(ys)[0]
    c_lo = _align4([xs.min()])[0]
    c_hi = _align4([xs.max()])[0]
    return [(r0, c) for c in range(c_lo, c_hi + 1, 4)]


def _occ(frame, top, left):
    return set(int(v) for v in np.unique(frame[max(0, top):top + 4, max(0, left):left + 4]))


def move(env, a):
    """Take one movement step; return True if the footprint actually moved.
    Terminal-safe: if the episode is over, does not step (stepping past the
    game-over transition returns an empty frame and would raise)."""
    if env.terminal():
        return False
    b = foot(env.frame())
    env.step(a)
    if env.terminal():
        return False
    return foot(env.frame()) != b


def in_progress(env, lvl0):
    """Loop-guard skill: the current level is still unsolved and playable.
    `levels_completed` is NON-zero on levels > 1, so guards must compare against
    the level captured at loop entry (lvl0), not test for zero."""
    return not env.terminal() and env.levels_completed <= lvl0


def nearest_to_foot(env, boxes):
    """The box (top,left) closest to the avatar footprint by Manhattan distance
    -- the cheapest next target to fetch."""
    fr, fc = foot(env.frame())
    return min(boxes, key=lambda b: abs(b[0] - fr) + abs(b[1] - fc))


def yield_to_helper(env, lvl0, a=1):
    """Hand the floor to the autonomous helper: idle (repeat the harmless action
    `a`) until the helper finishes the level or the episode ends. The helper acts
    on every avatar step, so idling lets it complete the remaining boxes."""
    while in_progress(env, lvl0):
        move(env, a)


def goto(env, tr, tc, blocked=(3, 4, 7), budget=150):
    """Collision-aware greedy navigation of the avatar footprint to (tr,tc),
    never stepping into a cell containing a blocked colour (boxes / wall)."""
    blocked = set(blocked)
    for _ in range(budget):
        r, cc = foot(env.frame())
        if (r, cc) == (tr, tc):
            return True
        prefs = []
        if r != tr:
            prefs.append(2 if tr > r else 1)
        if cc != tc:
            prefs.append(4 if tc > cc else 3)
        did = False
        for a in prefs:
            dr, dc = VEC[a]
            if blocked & _occ(env.frame(), r + dr, cc + dc):
                continue
            if move(env, a):
                did = True
                break
        if not did:  # detour: any safe move that reduces some coordinate
            for a in (1, 2, 3, 4):
                dr, dc = VEC[a]
                nr, nc = r + dr, cc + dc
                if nr < 0 or nc < 0 or nr > 60 or nc > 60:
                    continue
                if blocked & _occ(env.frame(), nr, nc):
                    continue
                reduces = (a in (1, 2) and r != tr) or (a in (3, 4) and cc != tc)
                if reduces and move(env, a):
                    did = True
                    break
            if not did:
                return False
    return False


def route(env, tr, tc, blocked=(2, 3, 4, 7), budget=200):
    """BFS pathfinder for the avatar footprint on the 4px grid, routing AROUND
    obstacles (unlike the greedy `goto`, which can get pinned when a box sits
    between it and the target). Re-plans after each executed step so it stays
    correct if the world shifts (e.g. a helper moves). The destination cell
    itself is allowed to be blocked (so you can walk up flush against a box)."""
    blocked = set(blocked)
    for _ in range(budget):
        f = env.frame()
        r, c = foot(f)
        if (r, c) == (tr, tc):
            return True
        prev = {(r, c): None}
        q = deque([(r, c)])
        found = False
        while q:
            cur = q.popleft()
            if cur == (tr, tc):
                found = True
                break
            for a, (dr, dc) in VEC.items():
                nr, nc = cur[0] + dr, cur[1] + dc
                if nr < 0 or nc < 0 or nr > 60 or nc > 60 or (nr, nc) in prev:
                    continue
                if (nr, nc) != (tr, tc) and (blocked & _occ(f, nr, nc)):
                    continue
                prev[(nr, nc)] = (cur, a)
                q.append((nr, nc))
        if not found:
            return False
        node, path = (tr, tc), []
        while prev[node] is not None:
            par, a = prev[node]
            path.append(a)
            node = par
        if not move(env, path[-1]):   # first step of the reconstructed path
            return False
    return False


def wall_col(frame, wall_color=2):
    """4-aligned left column of the tallest vertical barrier made of `wall_color`.
    Distinguishes a floor-to-ceiling wall from smaller same-coloured features
    (e.g. a container interior) by picking the busiest 4-aligned column band."""
    _, xs = np.where(frame == wall_color)
    if not len(xs):
        return None
    cnt = Counter((int(x) // 4) * 4 for x in xs)
    return max(cnt, key=lambda k: cnt[k])


def grab_from_left(env, box):
    """Approach `box` (top,left) from its LEFT, bump into it to face right, press 5
    to grab. The box then rides on the avatar's RIGHT side (carry offset (0,+4)) --
    the mirror of grab_from_below. Lets you deposit a box into a cell one column to
    your right, e.g. flush against a wall you cannot enter."""
    br, bc = box
    if not route(env, br, bc - 4):
        return False
    move(env, 4)      # bump right into the box (contact)
    env.step(5)       # grab
    return True


def relay_to_helper(env, box_color=4, wall_color=2):
    """Cooperative RELAY across an impassable wall. The avatar is trapped on one
    side; an autonomous HELPER on the far side ferries boxes from the wall into the
    goal but cannot reach boxes deeper on the avatar's side. So the avatar grabs
    each of its boxes and drops it flush against the wall (into the wall-band column
    the helper can pick from), waits for the helper to take it, then fetches the
    next. Once every box is relayed, idles so the helper finishes -> reward fires."""
    wc = wall_col(env.frame(), wall_color)
    lvl0 = env.levels_completed
    for _ in range(12):
        if not in_progress(env, lvl0):
            break
        boxes = [b for b in clusters(env.frame(), box_color) if b[1] < wc]
        if not boxes:
            break
        box = nearest_to_foot(env, boxes)
        row = box[0]
        if not grab_from_left(env, box):
            break
        place_carried(env, row, wc)           # drop flush against the wall
        for _ in range(45):                   # let the helper collect it
            if not in_progress(env, lvl0):
                break
            if not ({3, 4, 9} & _occ(env.frame(), row, wc)):
                break
            move(env, 1)
    yield_to_helper(env, lvl0)                # idle; helper finishes the level


def grab_from_below(env, box):
    """Navigate directly below `box` (top,left), touch it moving up, press 5 to
    grab. Afterwards the box rides with the avatar."""
    br, bc = box
    if not goto(env, br + 4, bc):
        return False
    move(env, 1)      # bump up into the box (contact)
    env.step(5)       # grab
    return True


def face_grab(env, a):
    """Grab the box adjacent in movement direction `a`. Grabbing requires FACING:
    the engine only attaches the sprite the nose points at, and a bump into the box
    (a blocked move) still sets the facing. The box then rides at the same relative
    offset it had at grab time (above / below / left / right)."""
    move(env, a)      # bump: sets rotation even though the box blocks the step
    env.step(5)       # grab


def carry_steps(env, a, n=1, tries=8):
    """While carrying, take `n` real steps in direction `a`, retrying transient
    blocks (e.g. a helper walking its cargo through the target cell). Rotation is
    frozen while carrying, so a failed step is safe to repeat."""
    done = 0
    for _ in range(n + tries):
        if done >= n or env.terminal():
            break
        if move(env, a):
            done += 1
    return done == n


def hoist_over_wall(env, stand, a, carry=1):
    """SMUGGLING skill: a CARRIED box may overlap static walls -- the engine checks
    the carried sprite's target cell only against occupied cells, not the wall set.
    A trapped avatar can therefore hand boxes across an enclosure wall: stand at
    `stand` (adjacent to the box, wall beyond), face-grab in direction `a`, take
    `carry` steps so the box comes to rest ON the wall band, and release. An
    outside helper can then pick the box off the wall and deliver it."""
    if not goto(env, *stand):
        return False
    face_grab(env, a)
    ok = carry_steps(env, a, carry)
    env.step(5)       # release (drops the box where it rides, wall included)
    return ok


def place_carried(env, trow, tcol):
    """With a box currently carried, drop it so its top-left lands at (trow,tcol).
    Calibrates the carry offset by probing a release on a clone, then positions
    the avatar precisely and releases."""
    c2 = env.clone()
    before = clusters(c2.frame(), 3)
    c2.step(5)
    new = [p for p in clusters(c2.frame(), 3) if p not in before]
    if not new:
        return False
    P = new[0]
    fr, fc = foot(env.frame())
    offr, offc = P[0] - fr, P[1] - fc
    if not goto(env, trow - offr, tcol - offc):
        return False
    env.step(5)
    return True


def carry_box_to_cell(env, box, cell):
    """Full skill: grab `box` and deposit it into target `cell` (row,col)."""
    if not grab_from_below(env, box):
        return False
    return place_carried(env, cell[0], cell[1])


def bin_rows(frame, interior=2):
    """4-aligned box-top rows that tile the container vertically (top..bottom)."""
    ys, _ = np.where(frame == interior)
    return _align4(ys)


def bin_columns(frame, interior=2):
    """4-aligned column lefts that tile the container horizontally."""
    _, xs = np.where(frame == interior)
    return _align4(xs)


def slot_occupied(frame, row, col):
    """A container slot holds a box iff a box border (dropped=3 / fresh=4) is in it."""
    return bool({3, 4} & _occ(frame, row, col))


def lowest_free_slot(frame, col, rows):
    """Lowest (largest-row) free slot in a column -- where a box dropped from
    above will rest without needing to stack on top of another. Returns row/None."""
    for r in sorted(rows, reverse=True):
        if not slot_occupied(frame, r, col):
            return r
    return None


def deliver_to_bin(env, box, col, row):
    """Grab `box` from below (so it rides above the avatar), carry it over `col`,
    descend until the box sits at `row`, then release. Works cleanly for the
    lowest free slot of a column because the avatar ends in the (free) cell just
    below the box."""
    if not grab_from_below(env, box):
        return False
    for _ in range(20):                       # slide to the target column
        c = foot(env.frame())[1]
        if c == col or not move(env, 4 if col > c else 3):
            break
    for _ in range(14):                       # descend until box reaches row
        if foot(env.frame())[0] >= row or not move(env, 2):
            break
    if not env.terminal():
        env.step(5)                           # release
    return True


def fill_bin_with_helper(env, box_color=4, interior=2, quota=2, park=(8, 0)):
    """Cooperative bin-filling: an autonomous HELPER also carries boxes to the
    container every turn but is too slow to finish alone in the step budget. The
    avatar personally delivers `quota` boxes into the lowest free slots (the cheap,
    always-reachable ones), then parks out of the way and idles so the helper --
    acting on every avatar move -- completes the remaining boxes. Win fires once
    ALL boxes rest on target cells."""
    rows = bin_rows(env.frame(), interior)
    cols = bin_columns(env.frame(), interior)

    def outside():
        # container-region boxes are excluded; a helper-carried box isn't colour-4
        rmin, rmax = min(rows) - 4, max(rows) + 8
        cmin, cmax = min(cols) - 4, max(cols) + 8
        return [b for b in clusters(env.frame(), box_color)
                if not (rmin <= b[0] <= rmax and cmin <= b[1] <= cmax)]

    lvl0 = env.levels_completed
    delivered = 0
    while delivered < quota and in_progress(env, lvl0):
        boxes = outside()
        if not boxes:
            break
        box = nearest_to_foot(env, boxes)
        # emptiest column whose bottom slot is free (avatar stands below it)
        cand = [(col, lowest_free_slot(env.frame(), col, rows)) for col in cols]
        cand = [(col, r) for col, r in cand if r is not None]
        if not cand:
            break
        col, row = min(cand, key=lambda cr: abs(cr[0] - box[1]))
        deliver_to_bin(env, box, col, row)
        delivered += 1

    goto(env, park[0], park[1], budget=20)       # step aside, then let helper finish
    yield_to_helper(env, lvl0)


def fill_targets(env, box_color=4, interior=2, key=lambda p: p[1]):
    """Higher-order leg: match each box (of `box_color`) to a target cell of the
    container (`interior`) under a shared sort `key` (default: column order),
    then carry each into place. The container reward fires once the last cell is
    filled. Boxes stay put until touched, so positions are read once up front."""
    cells = sorted(target_cells(env.frame(), interior), key=key)
    boxes = sorted(clusters(env.frame(), box_color), key=key)
    for box, cell in zip(boxes, cells):
        carry_box_to_cell(env, box, cell)

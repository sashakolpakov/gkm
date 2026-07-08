# Shared leg library: small, named, reusable skills.
# Players import from here; add a NEW leg only when no existing leg fits.

from collections import deque

import numpy as np

UP = 1
DOWN = 2
LEFT = 3
RIGHT = 4
USE = 5

_DIRS = {UP: (-1, 0), DOWN: (1, 0), LEFT: (0, -1), RIGHT: (0, 1)}


def repeat(env, action, times=1):
    for _ in range(times):
        if env.terminal():
            return
        env.step(action)


def up(env, times=1):
    repeat(env, UP, times)


def down(env, times=1):
    repeat(env, DOWN, times)


def left(env, times=1):
    repeat(env, LEFT, times)


def right(env, times=1):
    repeat(env, RIGHT, times)


def use(env, times=1):
    repeat(env, USE, times)


def follow_plan(env, *steps):
    for action, times in steps:
        repeat(env, action, times)


def move_and_use(env, action, steps=1):
    repeat(env, action, steps)
    use(env)


def ascend_and_use(env, steps=1):
    move_and_use(env, UP, steps)


def descend_and_use(env, steps=1):
    move_and_use(env, DOWN, steps)


def left_and_use(env, steps=1):
    move_and_use(env, LEFT, steps)


def right_and_use(env, steps=1):
    move_and_use(env, RIGHT, steps)


def service_upper_branch(
    env,
    outward_action,
    outward_steps,
    return_action,
    return_steps,
    *,
    descend_after_ascent=False,
):
    repeat(env, outward_action, outward_steps)
    ascend_and_use(env)
    if descend_after_ascent:
        down(env)
    repeat(env, return_action, return_steps)
    use(env)


def service_left_branch(env):
    service_upper_branch(env, LEFT, 4, RIGHT, 3)


def service_right_branch(env):
    service_upper_branch(env, RIGHT, 4, LEFT, 2, descend_after_ascent=True)


def enter_lower_right_lane(env):
    down(env, 3)
    right_and_use(env, 6)


def sweep_to_lower_left_switch(env):
    down(env)
    left(env, 7)
    descend_and_use(env)


def sweep_to_far_right_switch(env):
    up(env)
    right_and_use(env, 9)


def reset_on_lower_left_switch(env):
    left(env, 10)
    descend_and_use(env)


def nudge_to_center_lift(env):
    down(env)
    right(env)
    ascend_and_use(env)


def finish_right_exit(env):
    down(env, 3)
    right(env, 2)
    ascend_and_use(env)


def grab_push_release(
    env,
    face_action,
    push_action,
    push_steps=1,
    *,
    face_steps=1,
    depart_action=None,
    depart_steps=1,
):
    """Bump-face an adjacent box, attach with USE, push/carry it, release.

    ``face_steps=0`` grabs whatever is already adjacent (relay handoffs);
    ``depart_action`` optionally steps clear after the release.
    """
    repeat(env, face_action, face_steps)
    use(env)
    repeat(env, push_action, push_steps)
    use(env)
    if depart_action is not None and depart_steps > 0:
        repeat(env, depart_action, depart_steps)


def ferry_box(env, approach, face_action, push_action, push_steps=1):
    """Reposition along ``approach`` plan-steps, then grab-push-release a box."""
    follow_plan(env, *approach)
    grab_push_release(env, face_action, push_action, push_steps)


def yield_until_level_up(env, idle_action, cap=40):
    """Yield turns (with a harmless action) so helper agents can finish."""
    start = env.levels_completed
    for _ in range(cap):
        if env.terminal() or env.levels_completed > start:
            return
        env.step(idle_action)


def ferry_all_then_yield(env, specs, idle_action):
    """Ferry a batch of boxes, then yield for the helper (the recurring shape).

    ``specs`` is a list of ``(approach_plan, face_action, push_action, push_steps)``
    tuples -- one per box; each is handed to :func:`ferry_box`.  After the whole
    batch is placed in the helper-reachable handoff zone, cede turns with
    ``idle_action`` until the level counter increments.  This is the static
    (pre-planned) sibling of :func:`ferry_all_to_courier_then_yield`'s live-frame
    self-pacing loop -- both realise "ferry a batch, then yield for the helper".
    """
    for approach, face_action, push_action, push_steps in specs:
        ferry_box(env, approach, face_action, push_action, push_steps)
    yield_until_level_up(env, idle_action)


def relay_box_from_west(env, carry_steps, *, depart_action=None, depart_steps=1):
    """Grab the box already faced from the west and carry it east for a helper."""
    grab_push_release(
        env,
        RIGHT,
        RIGHT,
        carry_steps,
        face_steps=0,
        depart_action=depart_action,
        depart_steps=depart_steps,
    )


# --- Courier-handoff ferry machinery (grid-of-4 cells) --------------------
# Reusable skill for "sokoban/courier" levels (e.g. wa30 L5): the avatar is a
# 4x4 mover (colour 14 body + colour 0 head) that GRABs a 4x4 box (4-border,
# 9-core) by facing it and pressing USE, then translates it rigidly.  A helper
# agent (courier, colour 12) patrols a corridor and seats boxes the avatar
# leaves in its reach.  The win fires once every box is seated, so we shuttle
# each box to a small set of handoff cells (kept un-congested by rotating drop
# cells) and then yield turns for the courier to finish.

def _cells(env, box_border=4, box_core=9, courier=12, wall=5, G=16):
    """Coarse 16x16 cell view: avatar cell, head cell, box set, courier, walls."""
    f = np.asarray(env.frame())
    av = head = cour = None
    ys, xs = np.where(f == 14)
    if len(ys):
        av = (int(ys.min()) // 4, int(xs.min()) // 4)
    hy, hx = np.where(f == 0)
    if len(hy):
        head = (int(hy.min()) // 4, int(hx.min()) // 4)
    cy, cx = np.where(f == courier)
    if len(cy):
        cour = (int(cy.min()) // 4, int(cx.min()) // 4)
    boxes = set()
    walls = set()
    for R in range(G):
        for C in range(G):
            blk = f[R * 4:R * 4 + 4, C * 4:C * 4 + 4]
            u = set(int(v) for v in np.unique(blk))
            if box_core in u and (box_border in u or 3 in u) and len(u) <= 3:
                boxes.add((R, C))
            if int((blk == wall).sum()) >= 8:
                walls.add((R, C))
    return av, head, boxes, cour, walls


def _bfs_path(start, goal, blocked, G=16):
    if start == goal:
        return []
    q = deque([start])
    prev = {start: None}
    while q:
        cur = q.popleft()
        for a, (dr, dc) in _DIRS.items():
            nb = (cur[0] + dr, cur[1] + dc)
            if not (0 <= nb[0] < G and 0 <= nb[1] < G):
                continue
            if nb in prev:
                continue
            if nb in blocked and nb != goal:
                continue
            prev[nb] = (cur, a)
            if nb == goal:
                path = []
                n = nb
                while prev[n] is not None:
                    p, act = prev[n]
                    path.append(act)
                    n = p
                return path[::-1]
            q.append(nb)
    return None


def _bfs_pair(start_av, offset, goal_av, blocked, G=16):
    """Path for avatar so that avatar and box (avatar+offset) both stay clear."""
    def ok(a):
        b = (a[0] + offset[0], a[1] + offset[1])
        if not (0 <= a[0] < G and 0 <= a[1] < G and 0 <= b[0] < G and 0 <= b[1] < G):
            return False
        return a not in blocked and b not in blocked
    if start_av == goal_av:
        return []
    q = deque([start_av])
    prev = {start_av: None}
    while q:
        cur = q.popleft()
        for act, (dr, dc) in _DIRS.items():
            nb = (cur[0] + dr, cur[1] + dc)
            if nb in prev:
                continue
            if not ok(nb):
                continue
            prev[nb] = (cur, act)
            if nb == goal_av:
                path = []
                n = nb
                while prev[n] is not None:
                    p, ac = prev[n]
                    path.append(ac)
                    n = p
                return path[::-1]
            q.append(nb)
    return None


def _obstacles_courier(env):
    """(avatar_cell, blocked_set) for the courier world: walls, boxes, courier."""
    av, head, boxes, cour, walls = _cells(env)
    blocked = set(walls) | set(boxes)
    if cour:
        blocked.add(cour)
    return av, blocked


def _obstacles_grid(env):
    """(avatar_cell, blocked_set) for the fenced-goal world: walls and boxes."""
    av, boxes, walls = _grid_scan(env)
    return av, set(walls) | set(boxes)


def _walk_avatar_to(env, goal, read_obstacles, cap=60):
    """Walk the empty-handed avatar to ``goal``, replanning every step.

    ``read_obstacles(env) -> (avatar_cell, blocked_set)`` supplies live
    perception (so a moving helper is avoided as it shifts).  The goal and the
    avatar's own cell are freed before each BFS so a target that is transiently
    occupied never deadlocks us.  This is the single shared navigation skill
    behind :func:`_nav_to` (courier world) and :func:`_avatar_nav` (fenced-goal
    world); they differ only in which obstacle-reader they pass.
    """
    for _ in range(cap):
        if env.terminal():
            return False
        av, blocked = read_obstacles(env)
        if av == goal:
            return True
        blocked = set(blocked)
        blocked.discard(goal)
        blocked.discard(av)
        path = _bfs_path(av, goal, blocked)
        if not path:
            return False
        env.step(path[0])
    return read_obstacles(env)[0] == goal


def _nav_to(env, goal, cap=60):
    """Drive the empty avatar to a cell in the courier world (courier moves)."""
    return _walk_avatar_to(env, goal, _obstacles_courier, cap)


def _carry_pair(env, offset, goal_av, cap=60):
    for _ in range(cap):
        if env.terminal():
            return False
        av, head, boxes, cour, walls = _cells(env)
        if av == goal_av:
            return True
        carried = (av[0] + offset[0], av[1] + offset[1])
        blocked = set(walls)
        for b in boxes:
            if b != carried:
                blocked.add(b)
        if cour:
            blocked.add(cour)
        blocked.discard(goal_av)
        blocked.discard(av)
        path = _bfs_pair(av, offset, goal_av, blocked)
        if not path:
            return False
        before = av
        env.step(path[0])
        if _cells(env)[0] == before:
            continue  # transiently blocked by moving courier; retry
    return _cells(env)[0] == goal_av


def grab_and_deliver(env, box, drop):
    """Grab the 4x4 box at cell ``box`` and rigidly carry it onto cell ``drop``."""
    av, head, boxes, cour, walls = _cells(env)
    if box not in boxes:
        return False
    blocked = set(walls) | (set(boxes) - {box})
    if cour:
        blocked.add(cour)
    best = None
    for act, (dr, dc) in _DIRS.items():
        ac = (box[0] + dr, box[1] + dc)
        if not (0 <= ac[0] < 16 and 0 <= ac[1] < 16) or ac in blocked:
            continue
        p = _bfs_path(av, ac, blocked)
        if p is not None and (best is None or len(p) < best[0]):
            best = (len(p), ac)
    if best is None:
        return False
    ac = best[1]
    if not _nav_to(env, ac):
        return False
    face = {(-1, 0): UP, (1, 0): DOWN, (0, -1): LEFT, (0, 1): RIGHT}[
        (box[0] - ac[0], box[1] - ac[1])]
    env.step(face)   # bump-face: sets facing, highlights box
    env.step(USE)    # grab (border -> 0, box attaches rigidly)
    av2 = _cells(env)[0]
    offset = (box[0] - av2[0], box[1] - av2[1])
    goal_av = (drop[0] - offset[0], drop[1] - offset[1])
    if not _carry_pair(env, offset, goal_av):
        env.step(USE)  # release wherever we ended up
        return False
    env.step(USE)      # release on the drop cell
    return True


def _park_and_wait(env, moves=(UP, DOWN), setup=(RIGHT, RIGHT, DOWN, DOWN), cap=120):
    """Yield turns without grabbing: park clear of boxes, then wiggle in place."""
    for a in setup:
        if env.terminal() or env.levels_completed > _park_and_wait._base:
            return
        env.step(a)
    for i in range(cap):
        if env.terminal() or env.levels_completed > _park_and_wait._base:
            return
        env.step(moves[i % len(moves)])


def ferry_all_to_courier_then_yield(env, drops=((7, 10), (8, 10)),
                                    region_min_col=11, max_iters=60):
    """Shuttle every free box to rotating handoff cells, then yield for the courier.

    Rotating over several ``drops`` keeps the handoff un-congested so a single
    courier can seat all boxes inside the step budget.  Generalises level 4's
    "ferry a batch, then yield" to a self-pacing loop that reads the live frame.
    """
    _park_and_wait._base = env.levels_completed
    for _ in range(max_iters):
        if env.terminal() or env.levels_completed > _park_and_wait._base:
            return
        av, head, boxes, cour, walls = _cells(env)
        free = [b for b in boxes if b[1] >= region_min_col and b[0] < 15]
        at_drop = [d for d in drops if d in boxes]
        if not free and not at_drop:
            break  # everything is seated or in flight
        empty = [d for d in drops if d not in boxes and d != cour]
        if free and empty:
            free.sort(key=lambda b: abs(b[0] - av[0]) + abs(b[1] - av[1]))
            box = free[0]
            empty.sort(key=lambda d: abs(d[0] - box[0]) + abs(d[1] - box[1]))
            if not grab_and_deliver(env, box, empty[0]):
                for _ in range(3):
                    if env.terminal():
                        return
                    env.step(DOWN)
        else:
            # both drops busy: cede a turn so the courier can clear one
            _yield_one(env)
    _park_and_wait(env)


def _yield_one(env):
    av, head, boxes, cour, walls = _cells(env)
    best = None
    for a, (dr, dc) in _DIRS.items():
        nb = (av[0] + dr, av[1] + dc)
        if not (0 <= nb[0] < 16 and 0 <= nb[1] < 16):
            continue
        if nb in walls or nb in boxes or nb == cour:
            continue
        d = min([abs(nb[0] - b[0]) + abs(nb[1] - b[1]) for b in boxes], default=99)
        if best is None or d > best[0]:
            best = (d, a)
    env.step(best[1] if best else UP)


# --- Robust grab-carry-release delivery (fenced-goal sokoban) --------------
# Its own coarse-cell reader (``_grid_scan`` detects the box border/core pair,
# the avatar colour 14, and walls colour 5) feeds the SHARED pathing skills:
# navigation reuses ``_walk_avatar_to`` (via ``_avatar_nav``) and the rigid
# carry reuses ``_bfs_pair`` -- the exact same search the courier world uses --
# so no BFS/nav/carry code is written twice.  A box GRABbed via a bump-face +
# USE turns its border to the head colour and then translates rigidly with the
# avatar until USE releases it; releasing on a goal-container cell fills that
# slot.  ``carry_box_to`` tries every grab orientation and uses clone
# look-ahead so it also works when the only routable carry threads a one-wide
# wall gap (wa30 L6).

def _grid_scan(env, G=16):
    """Return (avatar_cell, box_cells:set, wall_cells:set)."""
    f = np.asarray(env.frame())
    av = None
    boxes = set()
    walls = set()
    for R in range(G):
        for C in range(G):
            blk = f[R * 4:R * 4 + 4, C * 4:C * 4 + 4]
            u = set(int(v) for v in np.unique(blk))
            if 14 in u:
                av = (R, C)
            if 9 in u and (4 in u or 3 in u) and 2 not in u:
                boxes.add((R, C))
            if int((blk == 5).sum()) >= 8:
                walls.add((R, C))
    return av, boxes, walls


def _grid_grabbed(env, G=16):
    """Cells of a currently-held box (head colour 0 wrapping a 9 core)."""
    f = np.asarray(env.frame())
    out = set()
    for R in range(G):
        for C in range(G):
            blk = f[R * 4:R * 4 + 4, C * 4:C * 4 + 4]
            u = set(int(v) for v in np.unique(blk))
            if 0 in u and 9 in u:
                out.add((R, C))
    return out


def _avatar_nav(env, goal, cap=60):
    """Walk the empty-handed avatar to ``goal`` in the fenced-goal world."""
    return _walk_avatar_to(env, goal, _obstacles_grid, cap)


def _rigid_carry(env, offset, goal_av, cap=120):
    """Carry the held box rigidly so the avatar reaches ``goal_av``.

    ``offset`` is (held_box - avatar); every move keeps both cells clear of
    walls and other boxes, replanning each step via the shared :func:`_bfs_pair`
    rigid-pair search (the same search :func:`_carry_pair` uses in the courier
    world; there the courier is an extra obstacle and a transient block is
    retried, whereas here a block means genuinely stuck).
    """
    for _ in range(cap):
        if env.terminal():
            return False
        av, boxes, walls = _grid_scan(env)
        if av == goal_av:
            return True
        held = _grid_grabbed(env)
        blocked = set(walls)
        for b in boxes:
            if b not in held:
                blocked.add(b)
        # Both the avatar cell and the held-box cell must start in-bounds/clear.
        bx = (av[0] + offset[0], av[1] + offset[1])
        if not (0 <= av[0] < 16 and 0 <= av[1] < 16
                and 0 <= bx[0] < 16 and 0 <= bx[1] < 16
                and av not in blocked and bx not in blocked):
            return False
        path = _bfs_pair(av, offset, goal_av, blocked)
        if not path:
            return False
        before = av
        env.step(path[0])
        if _grid_scan(env)[0] == before:
            return False
    return _grid_scan(env)[0] == goal_av


def carry_box_to(env, box, drop, cap=120):
    """Grab the 4x4 box at cell ``box`` and rigidly carry it onto ``drop``.

    Tries each grab orientation and uses a clone to confirm the carry is
    routable before committing, so it also handles carries that must thread a
    single-cell wall gap.  Returns True on a completed release at ``drop``.
    """
    av, boxes, walls = _grid_scan(env)
    if box not in boxes:
        return False
    faces = {(-1, 0): UP, (1, 0): DOWN, (0, -1): LEFT, (0, 1): RIGHT}
    for (dr, dc) in faces:
        ac = (box[0] + dr, box[1] + dc)
        if not (0 <= ac[0] < 16 and 0 <= ac[1] < 16):
            continue
        if ac in walls or ac in (boxes - {box}):
            continue
        face = faces[(box[0] - ac[0], box[1] - ac[1])]  # face from ac toward box
        trial = env.clone()
        if not _avatar_nav(trial, ac, cap):
            continue
        trial.step(face)
        trial.step(USE)
        held = _grid_grabbed(trial)
        if not held:
            continue
        tav = _grid_scan(trial)[0]
        offset = (box[0] - tav[0], box[1] - tav[1])
        goal_av = (drop[0] - offset[0], drop[1] - offset[1])
        probe = trial.clone()
        if not _rigid_carry(probe, offset, goal_av, cap):
            continue
        # feasible orientation -> commit on the real env
        if not _avatar_nav(env, ac, cap):
            return False
        env.step(face)
        env.step(USE)
        held = _grid_grabbed(env)
        if not held:
            return False
        rav = _grid_scan(env)[0]
        off = (box[0] - rav[0], box[1] - rav[1])
        gav = (drop[0] - off[0], drop[1] - off[1])
        if not _rigid_carry(env, off, gav, cap):
            env.step(USE)
            return False
        env.step(USE)
        return True
    return False


# --- Deliver-boxes-to-goals verbs + the neutralise-then-deliver combinator --
# Levels 6 and 7 share one skeleton: first NEUTRALISE a blocking autonomous
# agent (a self-mover that would otherwise steal boxes / keep the goal from
# staying filled), then DELIVER every arena box onto the goal-container cells
# with the shared :func:`carry_box_to` rigid carry.  Only the two halves vary:
# how the agent is neutralised (thread-a-gap + USE vs idle-until-frozen + USE),
# and how boxes are matched to goal cells (explicit pairs vs nearest-first).
# The delivery loop itself is written ONCE here as two tiny verbs; the shape is
# captured once as the higher-order :func:`neutralize_then_deliver`.


def deliver_pairs(env, moves):
    """Deliver boxes by an explicit plan: rigid-carry each box onto its drop.

    ``moves`` is a list of ``(box_cell, drop_cell)`` pairs, each handed to
    :func:`carry_box_to`.  Use when the box->goal assignment is known ahead of
    time (wa30 L6: the two east boxes go to specific west-container slots).
    """
    for box, drop in moves:
        carry_box_to(env, box, drop)


def fill_targets_nearest_first(env, targets):
    """Fill each goal cell in ``targets`` with the nearest still-free box.

    Re-reads the frame per target (boxes already sitting on a target are not
    re-grabbed), picks the closest remaining box, and rigid-carries it on.
    Use when any box may fill any slot (wa30 L7: fill the west 9-framed goal).
    """
    for drop in targets:
        av, boxes, walls = _grid_scan(env)
        free = [b for b in boxes if b not in targets]
        if not free:
            break
        free.sort(key=lambda b: abs(b[0] - drop[0]) + abs(b[1] - drop[1]))
        carry_box_to(env, free[0], drop)


def neutralize_then_deliver(env, neutralize, deliver):
    """Higher-order shape shared by wa30 L6 & L7: disable the blocking agent,
    then deliver boxes into the goal container.

    ``neutralize(env)`` removes/settles the autonomous agent so placements
    stick; ``deliver(env)`` then carries every box onto its goal cell (via
    :func:`deliver_pairs` or :func:`fill_targets_nearest_first`).  With the
    agent gone a filled container is the win.
    """
    neutralize(env)
    deliver(env)


def clear_agent_then_deliver(env, approach, moves):
    """wa30-L6 shape: thread the wall gap to the parked self-mover, USE to
    clear it, then ferry each arena box into the goal container.

    ``approach`` is a follow_plan tuple-list landing the avatar beside the
    mover (the trailing USE removes it); ``moves`` is a list of ``(box, drop)``
    cell pairs.  A thin :func:`neutralize_then_deliver` composition.
    """
    def neutralize(e):
        follow_plan(e, *approach)
        use(e)

    neutralize_then_deliver(env, neutralize, lambda e: deliver_pairs(e, moves))


# --- Uncatchable self-mover: freeze it, then clear and deliver (wa30 L7) ----
# A patrolling self-mover (colour ``mover``) relentlessly hauls every loose
# 4x4 box to ITS store, so the avatar can never keep a box in the goal
# container while the mover is active -- and USE has no effect on a *moving*
# mover.  But when the mover grabs a box it cannot seat it stalls (behavioural
# freeze); a stalled mover CAN be cleared with USE (the L6 clear mechanic).  So
# we idle until the mover is stationary, clear it (freeing the box it held),
# then -- with no mover left to steal -- rigidly carry every box into the goal
# container.  This composes the existing idle/clear/carry skills.

def _mover_cell(env, mover):
    f = np.asarray(env.frame())
    ys, xs = np.where(f == mover)
    if not len(ys):
        return None
    return (int(ys.min()) // 4, int(xs.min()) // 4)


def idle_until_mover_frozen(env, idle_action, mover=15, settle=5,
                            min_step=16, cap=45):
    """Cede turns (harmless ``idle_action``) until the self-mover stalls.

    The mover is 'frozen' once its cell is unchanged for ``settle`` consecutive
    turns after ``min_step`` (early transient stalls, e.g. bouncing off a box,
    are ignored).  Returns the frozen cell (or None if the mover is gone)."""
    prev = None
    stable = 0
    for i in range(cap):
        if env.terminal():
            return _mover_cell(env, mover)
        p = _mover_cell(env, mover)
        if p is None:
            return None
        stable = stable + 1 if p == prev else 0
        prev = p
        if stable >= settle and i > min_step:
            return p
        env.step(idle_action)
    return _mover_cell(env, mover)


def clear_frozen_mover(env, mover=15):
    """Step beside a stalled self-mover and USE to remove it (frees its box).

    Tries each adjacent cell (preferring from below); clones a probe walk first
    so a blocked side is skipped.  Returns True once the mover colour is gone.
    """
    p = _mover_cell(env, mover)
    if p is None:
        return True
    faces = [((1, 0), UP), ((0, -1), RIGHT), ((0, 1), LEFT), ((-1, 0), DOWN)]
    for (dr, dc), face in faces:
        adj = (p[0] + dr, p[1] + dc)
        if not (0 <= adj[0] < 16 and 0 <= adj[1] < 16):
            continue

        def block(e, _mp=p):
            av, boxes, walls = _grid_scan(e)
            blk = set(walls) | set(boxes)
            mc = _mover_cell(e, mover)
            if mc is not None:
                blk.add(mc)
            return av, blk

        trial = env.clone()
        if not _walk_avatar_to(trial, adj, block, 60):
            continue
        if not _walk_avatar_to(env, adj, block, 60):
            continue
        env.step(face)
        env.step(USE)
        if _mover_cell(env, mover) is None:
            return True
    return _mover_cell(env, mover) is None


def clear_frozen_mover_then_fill(env, targets, idle_action=UP, mover=15):
    """wa30-L7 shape: neutralise the uncatchable hauler, then fill the goal.

    ``targets`` are the goal-container cells to fill.  Neutralising means idle
    until the mover stalls then clear it with USE; delivery is nearest-first so
    any box may fill any slot.  A thin :func:`neutralize_then_deliver`
    composition sharing L6's delivery machinery.
    """
    def neutralize(e):
        idle_until_mover_frozen(e, idle_action, mover)
        clear_frozen_mover(e, mover)

    neutralize_then_deliver(
        env, neutralize, lambda e: fill_targets_nearest_first(e, targets))



# --- Two-socket world with same-speed self-mover "stealers" (wa30 L8) --------
# Two socket containers (9-frame / 2-core) each have a courier (colour 12) that
# auto-seats loose boxes -- but each also has a roaming self-mover STEALER
# (colour 15, one per band) that pulls boxes back OUT of its socket, so a
# courier can never finish while its stealer roams.  A stealer moves at the
# avatar's own speed, so it cannot be tail-chased; but its bounce loop turns it
# into a corner where a facing USE removes it PERMANENTLY.  Remove both
# stealers, then the two couriers (helped by a few avatar deliveries) drain
# both sockets and the level completes.

def _movers(env, color):
    """Coarse 16x16 cells occupied by a self-mover of ``color``."""
    f = np.asarray(env.frame())
    ys, xs = np.where(f == color)
    return sorted(set((int(y) // 4, int(x) // 4) for y, x in zip(ys, xs)))


def chase_and_clear(env, color, band_pred, approach=None, cap=55):
    """Pursue a same-speed self-mover of ``color`` and USE it away.

    ``band_pred(cell)`` selects which mover instance to hunt (e.g. one band);
    ``approach`` is an optional cell to navigate to first (enter that band).
    Pursuit is box/wall-aware (:func:`_bfs_path`) with a greedy distance-
    reducing fallback when boxes block the shortest route; a facing USE fires
    whenever we are orthogonally adjacent, which removes the mover for good.
    """
    if approach is not None:
        _avatar_nav(env, approach, cap=20)
    for _ in range(cap):
        if env.terminal():
            return
        ms = [m for m in _movers(env, color) if band_pred(m)]
        if not ms:
            return
        av, boxes, walls = _grid_scan(env)
        ms.sort(key=lambda m: abs(av[0] - m[0]) + abs(av[1] - m[1]))
        m = ms[0]
        if abs(av[0] - m[0]) + abs(av[1] - m[1]) == 1:
            env.step(USE)
            continue
        blocked = (set(walls) | set(boxes)) - {m}
        p = _bfs_path(av, m, blocked)
        if p:
            env.step(p[0])
            continue
        best = None
        for act, (dr, dc) in _DIRS.items():
            nb = (av[0] + dr, av[1] + dc)
            if not (0 <= nb[0] < 16 and 0 <= nb[1] < 16):
                continue
            if nb in walls or nb in boxes:
                continue
            dd = abs(nb[0] - m[0]) + abs(nb[1] - m[1])
            if best is None or dd < best[0]:
                best = (dd, act)
        env.step(best[1] if best else USE)


def fill_dual_sockets(env, top_cells, bot_cells, split_row,
                      cap=200, idle_action=UP):
    """Drain two socket containers: deliver the box nearest the avatar into an
    open cell of the socket for its band, ceding turns to the couriers when no
    box is currently deliverable.  Runs until the level counter increments or
    the step budget (colour-7 bar) runs out.

    ``split_row`` decides a box's target socket (row < split_row -> top).
    A short idle fallback lets the couriers seat/settle boxes the avatar can't
    reach, and repeated idle-with-no-progress ends in a final yield.
    """
    base = env.levels_completed
    tset = set(top_cells) | set(bot_cells)
    stuck = 0
    for _ in range(cap):
        if env.terminal() or env.levels_completed > base:
            return
        av, boxes, walls = _grid_scan(env)
        free = [b for b in boxes if b not in tset]
        open_top = [t for t in top_cells if t not in boxes]
        open_bot = [t for t in bot_cells if t not in boxes]

        def targets(b):
            return open_top if b[0] < split_row else open_bot

        free.sort(key=lambda b: abs(b[0] - av[0]) + abs(b[1] - av[1]))
        made = False
        for b in free[:8]:
            ts = sorted(targets(b),
                        key=lambda t: abs(b[0] - t[0]) + abs(b[1] - t[1]))[:2]
            for t in ts:
                if carry_box_to(env, b, t):
                    made = True
                    break
            if made:
                break
        if made:
            stuck = 0
            continue
        # nothing deliverable right now: let the couriers work a few turns
        stuck += 1
        for _ in range(5):
            if env.terminal() or env.levels_completed > base:
                return
            env.step(idle_action)
        if stuck > 5:
            break
    # final yield: couriers finish the last seatings within the budget
    while not env.terminal() and env.levels_completed <= base:
        env.step(idle_action)


def clear_stealers_then_fill_dual(env, bands, top_cells, bot_cells, split_row,
                                  mover=15):
    """wa30-L8 shape: remove the per-band roaming stealers, then let the two
    couriers (plus a few avatar deliveries) drain both socket containers.

    ``bands`` is a list of ``(band_pred, approach_cell)`` handed to
    :func:`chase_and_clear` one per stealer; then :func:`fill_dual_sockets`
    finishes.  Like L6 & L7 this is the neutralise-then-deliver shape, so it
    routes through the shared :func:`neutralize_then_deliver` combinator: the
    neutralise half clears every band's stealer, the deliver half drains both
    sockets.  A thin composition over the shared grab/carry/nav primitives.
    """
    def neutralize(e):
        for band_pred, approach in bands:
            if e.terminal():
                return
            chase_and_clear(e, mover, band_pred, approach)

    neutralize_then_deliver(
        env, neutralize,
        lambda e: fill_dual_sockets(e, top_cells, bot_cells, split_row))


def ambush_and_clear(env, cell, face_action, mover=15, cap=40):
    """Park on ``cell`` and USE the roaming self-mover away as it walks into
    the faced neighbour cell (wa30 L9: the maze stealer surfaces at a fixed
    corridor mouth).

    Cheaper than :func:`chase_and_clear` when pursuit is impossible or too
    expensive (tight step budget, comb maze): navigate once to the ambush
    cell, then wait; a USE alone is a safe idle, and ``face_action`` is only
    pressed when the mover occupies the faced cell (so it cannot move us off
    the ambush spot).  Returns True once no ``mover`` cells remain.
    """
    _avatar_nav(env, cell)
    dr, dc = _DIRS[face_action]
    tgt = (cell[0] + dr, cell[1] + dc)
    for _ in range(cap):
        if env.terminal():
            return False
        ms = _movers(env, mover)
        if not ms:
            return True
        if tgt in ms:
            env.step(face_action)
        env.step(USE)
    return not _movers(env, mover)

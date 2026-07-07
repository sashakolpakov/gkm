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


def _nav_to(env, goal, cap=60):
    """Drive the empty avatar to a cell, replanning each step (courier moves)."""
    for _ in range(cap):
        if env.terminal():
            return False
        av, head, boxes, cour, walls = _cells(env)
        if av == goal:
            return True
        blocked = set(walls) | set(boxes)
        if cour:
            blocked.add(cour)
        blocked.discard(goal)
        blocked.discard(av)
        path = _bfs_path(av, goal, blocked)
        if not path:
            return False
        env.step(path[0])
    return _cells(env)[0] == goal


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

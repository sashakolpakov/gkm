import numpy as np
import collections

# Action codes (discovered by experiment):
#  1=up  2=down  3=left  4=right  5=interact(attach/release)
UP, DOWN, LEFT, RIGHT, ACT = 1, 2, 3, 4, 5
DIRS = {UP: (-1, 0), DOWN: (1, 0), LEFT: (0, -1), RIGHT: (0, 1)}
N = 16  # macro grid is 16x16 (each macro cell = 4x4 pixels)


def macro_center(f, R, C):
    return f[4 * R + 1:4 * R + 3, 4 * C + 1:4 * C + 3]


def avatar_cell(f):
    ys, xs = np.where(f == 14)
    if len(ys) == 0:
        return None
    return (int(ys.min()) // 4, int(xs.min()) // 4)


def loose_carriers(f, contset):
    """Carriers sitting OUTSIDE the container: macro centre is colour 9."""
    out = []
    for R in range(N):
        for C in range(N):
            if (R, C) in contset:
                continue
            if np.all(macro_center(f, R, C) == 9):
                out.append((R, C))
    return out


def wall_columns(f):
    """A dividing wall is a near-full-height column of colour 2 (textured).
    Return the set of such column indices."""
    cols = set()
    for C in range(N):
        cnt = sum(1 for R in range(N) if np.any(macro_center(f, R, C) == 2))
        if cnt >= 12:
            cols.add(C)
    return cols


def container_cells(f):
    """Every macro cell whose centre is the container interior colour (2)
    (excluding any dividing wall column), plus the bounding box of those cells
    (so the colour-9 border is included)."""
    wcols = wall_columns(f)
    twos = [(R, C) for R in range(N) for C in range(N)
            if np.all(macro_center(f, R, C) == 2) and C not in wcols]
    if not twos:
        return set(), set()
    r0 = min(r for r, _ in twos); r1 = max(r for r, _ in twos)
    c0 = min(c for _, c in twos); c1 = max(c for _, c in twos)
    bbox = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}
    return set(twos), bbox


def empty_slots(f, twos):
    return [s for s in twos if np.all(macro_center(f, s[0], s[1]) == 2)]


def helper_cells(f):
    out = set()
    for R in range(N):
        for C in range(N):
            if np.any(f[4 * R:4 * R + 4, 4 * C:4 * C + 4] == 12):
                out.add((R, C))
    return out


def walls_of(f):
    walls = set()
    for R in range(N):
        for C in range(N):
            if np.any(f[4 * R:4 * R + 4, 4 * C:4 * C + 4] == 7):
                walls.add((R, C))
    return walls


def bfs(start, goal, free_fn):
    if start == goal:
        return []
    q = collections.deque([start]); prev = {start: (None, None)}
    while q:
        cur = q.popleft()
        for a, (dr, dc) in DIRS.items():
            nxt = (cur[0] + dr, cur[1] + dc)
            if nxt in prev or not (0 <= nxt[0] < N and 0 <= nxt[1] < N):
                continue
            if not free_fn(nxt):
                continue
            prev[nxt] = (cur, a)
            if nxt == goal:
                acts = []; node = nxt
                while prev[node][0] is not None:
                    p, a2 = prev[node]; acts.append(a2); node = p
                return acts[::-1]
            q.append(nxt)
    return None


def bfs_carry(start, goal, offset, free_fn):
    odr, odc = offset

    def unit_ok(av):
        if not (0 <= av[0] < N and 0 <= av[1] < N):
            return False
        car = (av[0] + odr, av[1] + odc)
        if not (0 <= car[0] < N and 0 <= car[1] < N):
            return False
        return free_fn(av) and free_fn(car)

    if start == goal:
        return []
    q = collections.deque([start]); prev = {start: (None, None)}
    while q:
        cur = q.popleft()
        for a, (dr, dc) in DIRS.items():
            nxt = (cur[0] + dr, cur[1] + dc)
            if nxt in prev or not unit_ok(nxt):
                continue
            prev[nxt] = (cur, a)
            if nxt == goal:
                acts = []; node = nxt
                while prev[node][0] is not None:
                    p, a2 = prev[node]; acts.append(a2); node = p
                return acts[::-1]
            q.append(nxt)
    return None


# (attach_cell_delta, face_action_into_carrier, carrier_offset_from_avatar)
SIDES = [
    ((0, 1), LEFT, (0, -1)),   # avatar right of carrier -> push left
    ((0, -1), RIGHT, (0, 1)),  # avatar left of carrier -> push right
    ((1, 0), UP, (-1, 0)),     # avatar below carrier -> push up
    ((-1, 0), DOWN, (1, 0)),   # avatar above carrier -> push down
]

AWAY = {LEFT: RIGHT, RIGHT: LEFT, UP: DOWN, DOWN: UP}


def plan_delivery(f, carrier, av, loose_set, contset, bbox, walls):
    """Return (nav, face, carry, away) to bring `carrier` into an empty slot."""
    twos = contset
    emp = empty_slots(f, twos)
    if not emp:
        return None
    block = (loose_set - {carrier}) | walls
    for slot in sorted(emp, key=lambda s: abs(s[0] - carrier[0]) + abs(s[1] - carrier[1])):
        for delta, face, off in SIDES:
            acell = (carrier[0] + delta[0], carrier[1] + delta[1])
            if not (0 <= acell[0] < N and 0 <= acell[1] < N):
                continue
            if acell in block or acell in bbox:
                continue

            def nav_free(c):
                return (0 <= c[0] < N and 0 <= c[1] < N and c not in block
                        and c not in bbox and c != carrier)

            nav = bfs(av, acell, nav_free)
            if nav is None:
                continue
            goal_av = (slot[0] - off[0], slot[1] - off[1])
            if not (0 <= goal_av[0] < N and 0 <= goal_av[1] < N):
                continue

            def carry_free(c):
                return 0 <= c[0] < N and 0 <= c[1] < N and c not in block

            carry = bfs_carry(acell, goal_av, off, carry_free)
            if carry is None:
                continue
            return nav, face, carry, AWAY[face]
    return None


def solve(env):
    real_used = [0]

    def step(a):
        if env.terminal():
            return None
        real_used[0] += 1
        return env.step(a)

    cur_level = -1
    stuck = 0
    twos = set()
    bbox = set()
    while not env.terminal():
        f = env.frame()
        lvl = env.levels_completed
        if lvl != cur_level:
            # Freeze the container region at level start: once a slot is filled it
            # turns colour 9 and stops looking like the container, so we must not
            # re-detect it every frame.
            twos, bbox = container_cells(f)
            cur_level = lvl
            stuck = 0

        if not twos:
            twos, bbox = container_cells(f)
        if not twos:
            break
        walls = walls_of(f)
        helper_present = bool(helper_cells(f))
        excl = bbox
        if helper_present:
            # Exclude a 1-cell margin around the container so we don't try to grab
            # a carrier the helper is in the middle of delivering.
            rr = [r for r, _ in bbox]; ccs = [c for _, c in bbox]
            excl = {(r, c) for r in range(min(rr) - 1, max(rr) + 2)
                    for c in range(min(ccs) - 1, max(ccs) + 2)}
        loose = loose_carriers(f, excl)
        av = avatar_cell(f)
        if av is None:
            break

        # ---- RELAY MODE (a dividing wall separates us from the container) ----
        wcols = wall_columns(f)
        if wcols and helper_present:
            wall_c = sorted(wcols)[len(wcols) // 2]
            cont_c = sum(c for _, c in twos) / len(twos)
            if (av[1] - wall_c) * (cont_c - wall_c) < 0:
                # We and the container are on opposite sides of the wall.
                if av[1] < wall_c:
                    side_lo, side_hi = 0, wall_c
                    face, off, away = RIGHT, (0, 1), LEFT
                else:
                    side_lo, side_hi = wall_c + 1, N
                    face, off, away = LEFT, (0, -1), RIGHT
                # carriers on our side (centre 9, not on the wall column)
                mine = []
                for R in range(N):
                    for C in range(side_lo, side_hi):
                        if C == wall_c:
                            continue
                        if np.all(macro_center(f, R, C) == 9):
                            mine.append((R, C))
                if not mine:
                    step(away)  # nothing to relay; tick the helper
                    continue
                mine.sort(key=lambda c: abs(c[0] - av[0]) + abs(c[1] - av[1]))
                done = False
                for car in mine:
                    appr = (car[0], car[1] - off[1])
                    if not (side_lo <= appr[1] < side_hi):
                        continue
                    block = set(mine) - {car}

                    def nav_free(c):
                        return (0 <= c[0] < N and side_lo <= c[1] < side_hi
                                and c[1] != wall_c and c not in block and c != car)

                    if not nav_free(appr):
                        continue
                    nav = bfs(av, appr, nav_free)
                    if nav is None:
                        continue
                    for a in nav:
                        step(a)
                        if env.terminal():
                            return
                    step(face)   # turn into carrier (blocked -> just turns)
                    step(ACT)    # attach
                    # push toward the wall until the avatar is blocked by it
                    for _ in range(N):
                        before = avatar_cell(env.frame())
                        step(face)
                        if env.terminal():
                            return
                        if avatar_cell(env.frame()) == before:
                            break
                    step(ACT)    # release the carrier onto the wall
                    if env.terminal() or env.levels_completed != lvl:
                        done = True
                        break
                    step(away)   # step off so we don't re-grab it
                    done = True
                    break
                if not done:
                    step(away)
                continue

        if not loose or not empty_slots(f, twos):
            # nothing left for us to deliver. If a helper is on the board, keep
            # ticking it until it finishes (each of our moves advances it);
            # otherwise there is nothing more to do.
            if helper_present:
                step(UP)
                continue
            stuck += 1
            if stuck > 5:
                break
            step(UP)
            continue

        loose_set = set(loose)
        cr = sum(r for r, _ in twos) / len(twos)
        cc = sum(c for _, c in twos) / len(twos)

        plan = None
        if helper_present:
            # Cooperative level: a helper delivers nearest-first, so WE take the
            # carriers farthest from the container to complement it.
            loose.sort(key=lambda c: -(abs(c[0] - cr) + abs(c[1] - cc)))
            for carrier in loose:
                plan = plan_delivery(f, carrier, av, loose_set, twos, bbox, walls)
                if plan:
                    break
        else:
            # Solo level: pick the closest carrier->slot pairing (fills an
            # open-ended container in a reachable order).
            emp = empty_slots(f, twos)
            cand = []
            for carrier in loose:
                for slot in emp:
                    cand.append((abs(carrier[0] - slot[0]) + abs(carrier[1] - slot[1]),
                                 carrier))
            cand.sort(key=lambda x: x[0])
            seen = []
            for _, carrier in cand:
                if carrier in seen:
                    continue
                seen.append(carrier)
                plan = plan_delivery(f, carrier, av, loose_set, twos, bbox, walls)
                if plan:
                    break

        if plan is None:
            if helper_present:
                step(UP)
                continue
            stuck += 1
            if stuck > 5:
                break
            step(UP)
            continue
        stuck = 0

        nav, face, carry, away = plan
        for a in nav:
            step(a)
            if env.terminal():
                return
        step(face)   # orient into carrier (blocked move just turns)
        step(ACT)    # attach
        for a in carry:
            step(a)
            if env.terminal():
                return
        step(ACT)    # release into slot
        if env.terminal() or env.levels_completed != lvl:
            continue  # delivery finished the level; don't spend a move in the next
        step(away)   # step off so we don't re-grab it

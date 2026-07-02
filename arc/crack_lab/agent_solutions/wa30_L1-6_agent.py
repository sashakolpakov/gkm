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


# ===================== L4+ : trapped-avatar ejection =====================
# On later levels the avatar is sealed inside a room of textured colour-2 walls
# (tag bnzklblgdk).  It cannot leave, but a CARRIED carrier can be pushed ONTO
# such a wall cell (the engine only blocks the *avatar* from wall cells, not the
# object it carries).  Autonomous colour-12 helpers roam the outside; each one
# picks up a carrier that is next to it and carries it to the NEAREST container.
# So our whole job is: eject every inside carrier onto a boundary wall next to a
# helper-reachable EMPTY container, then idle so the helpers deliver.

def _block(f, R, C):
    return f[4 * R:4 * R + 4, 4 * C:4 * C + 4]


def carriers_all(f):
    """Every carrier (macro centre is colour 9), delivered or not."""
    return {(R, C) for R in range(N) for C in range(N)
            if np.all(macro_center(f, R, C) == 9)}


def wallA_cells(f):
    """Textured colour-2 walls (pmargquscu): block 2s mixed with background 1s,
    no carrier/target/agent colours."""
    out = set()
    for R in range(N):
        for C in range(N):
            s = set(_block(f, R, C).flatten().tolist())
            if 2 in s and 1 in s and not ({9, 4, 14, 12, 5} & s):
                out.add((R, C))
    return out


def cont2_cells(f):
    """Container cells (9-border/2-interior boxes) not currently holding a
    carrier.  A carrier on a container shows border 4/5/0, so its block is not a
    subset of {2,9}."""
    wa = wallA_cells(f)
    out = set()
    for R in range(N):
        for C in range(N):
            if (R, C) in wa:
                continue
            s = set(_block(f, R, C).flatten().tolist())
            if 2 in s and s <= {2, 9}:
                out.add((R, C))
    return out


def colour5_cells(f):
    return {(R, C) for R in range(N) for C in range(N)
            if np.all(_block(f, R, C) == 5)}


def _bfs_free(start, goals, blocked):
    if start in goals:
        return []
    q = collections.deque([start]); prev = {start: (None, None)}
    while q:
        cur = q.popleft()
        for a, (dr, dc) in DIRS.items():
            nx = (cur[0] + dr, cur[1] + dc)
            if nx in prev or not (0 <= nx[0] < N and 0 <= nx[1] < N):
                continue
            if nx in blocked:
                continue
            prev[nx] = (cur, a)
            if nx in goals:
                acts = []; nd = nx
                while prev[nd][0] is not None:
                    p, a2 = prev[nd]; acts.append(a2); nd = p
                return acts[::-1]
            q.append(nx)
    return None


def _walk_dist(src, blocked):
    """Manhattan-ish shortest-path distances from src over free cells."""
    dist = {src: 0}; q = collections.deque([src])
    while q:
        cur = q.popleft()
        for dr, dc in DIRS.values():
            nx = (cur[0] + dr, cur[1] + dc)
            if 0 <= nx[0] < N and 0 <= nx[1] < N and nx not in blocked and nx not in dist:
                dist[nx] = dist[cur] + 1
                q.append(nx)
    return dist


def avatar_region(f, av):
    """Cells the avatar can walk to (blocked by walls / colour-5 / carriers)."""
    blocked = wallA_cells(f) | colour5_cells(f) | carriers_all(f)
    reg = {av}; q = collections.deque([av])
    while q:
        cur = q.popleft()
        for dr, dc in DIRS.values():
            nx = (cur[0] + dr, cur[1] + dc)
            if 0 <= nx[0] < N and 0 <= nx[1] < N and nx not in blocked and nx not in reg:
                reg.add(nx); q.append(nx)
    return reg


def _plan_eject(f, carrier):
    """For `carrier`, over the 4 push directions, return list of
    (push_cost, dir, approach_cell, wall_cell, outside_neighbor)."""
    wa = wallA_cells(f); c5 = colour5_cells(f); cars = carriers_all(f)
    opts = []
    for d, (dr, dc) in DIRS.items():
        r, c = carrier; steps = 0; ok = True
        while True:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < N and 0 <= nc < N):
                ok = False; break
            if (nr, nc) in wa:
                break                       # carrier lands here (on the wall)
            if (nr, nc) in (cars - {carrier}) or (nr, nc) in c5:
                ok = False; break
            r, c = nr, nc; steps += 1
            if steps > 6:
                ok = False; break
        if not ok:
            continue
        wall = (r + dr, c + dc)
        appr = (carrier[0] - dr, carrier[1] - dc)
        if not (0 <= appr[0] < N and 0 <= appr[1] < N):
            continue
        if appr in wa or appr in c5 or appr in (cars - {carrier}):
            continue
        outside = (wall[0] + dr, wall[1] + dc)
        opts.append((steps + 1, d, appr, wall, outside))
    return opts


def _assign_targets(f, av):
    """Global 1:1 matching of inside carriers to distinct empty containers.
    Carriers already OUTSIDE the room reserve their own nearest container (a
    helper will deliver those), so we don't send an inside carrier to a box that
    is already spoken for.  Returns {carrier: (dir, approach)}."""
    reg = avatar_region(f, av)
    cars = carriers_all(f)
    conts = cont2_cells(f)
    inside = [c for c in cars if c not in conts and any(
        (c[0] + dr, c[1] + dc) in reg for dr, dc in DIRS.values())]
    outside = [c for c in cars if c not in conts and c not in inside]
    outside_blocked = wallA_cells(f) | colour5_cells(f)
    empties = set(conts)
    # reserve a container for each already-outside carrier (nearest by walk)
    for c in outside:
        dist = _walk_dist(c, outside_blocked)
        cand = [(dist[e], e) for e in empties if e in dist]
        if cand:
            empties.discard(min(cand)[1])
    # build (cost, carrier, dir, approach, container) candidates
    cands = []
    for c in inside:
        for cost, d, appr, wall, outn in _plan_eject(f, c):
            if appr not in reg:
                continue
            dist = _walk_dist(outn, outside_blocked)
            for e in empties:
                if e in dist:
                    cands.append((dist[e] + cost, c, d, appr, e))
    cands.sort(key=lambda t: t[0])
    used_c, used_e, assign = set(), set(), {}
    for _, c, d, appr, e in cands:
        if c in used_c or e in used_e:
            continue
        used_c.add(c); used_e.add(e); assign[c] = (d, appr)
    return assign


def eject_controller(env, step, lvl):
    """Trapped-avatar loop: eject every inside carrier onto a boundary wall next
    to its assigned empty container, then idle so helpers deliver."""
    f = env.frame(); av = avatar_cell(f)
    if av is None:
        return
    # plan the whole assignment ONCE (recomputing would let already-ejected wall
    # carriers reserve the containers the remaining inside carriers still need).
    assign = _assign_targets(f, av)
    todo = dict(assign)
    while todo and not env.terminal() and env.levels_completed == lvl:
        f = env.frame(); av = avatar_cell(f)
        if av is None:
            return
        cars = carriers_all(f)
        reg = avatar_region(f, av)
        # a carrier is still "to eject" only while it sits at its planned cell
        pending = {c: v for c, v in todo.items() if c in cars}
        for c in list(todo):
            if c not in cars:
                del todo[c]
        # pick a pending carrier whose approach cell is free & reachable now
        ready = [(abs(c[0] - av[0]) + abs(c[1] - av[1]), c, d, appr)
                 for c, (d, appr) in pending.items()
                 if appr in reg and appr not in cars]
        if not ready:
            _idle(env, step, av, f)
            continue
        ready.sort()
        _, carrier, d, appr = ready[0]
        del todo[carrier]
        nav = _bfs_free(av, {appr},
                        wallA_cells(f) | colour5_cells(f) | cars)
        if nav is None:
            _idle(env, step, av, f)
            continue
        for a in nav:
            step(a)
            if env.terminal() or env.levels_completed != lvl:
                return
        step(d)          # face the carrier (blocked move just turns)
        step(ACT)        # attach
        for _ in range(8):
            before = avatar_cell(env.frame())
            step(d)
            if env.terminal() or env.levels_completed != lvl:
                return
            if avatar_cell(env.frame()) == before:
                break    # carrier now rests on the wall
        step(ACT)        # release
        step(AWAY[d])    # step off
    # everything ejected -> idle and let the helpers finish delivering
    while not env.terminal() and env.levels_completed == lvl:
        f = env.frame(); av = avatar_cell(f)
        if av is None:
            return
        _idle(env, step, av, f)


# ===================== L6+ : solo delivery with adversary =====================
# Some levels have NO helper and a colour-15 sprite that STEALS carriers and drags
# them to decoy solid-2 blocks.  The avatar must destroy that adversary (ACT while
# facing it) and then carry every loose carrier to the real container itself.  A
# moving colour-15 desyncs static planning, so we VERIFY every candidate action
# sequence on a clone (the real engine) before committing.

def col15_cells(f):
    return {(R, C) for R in range(N) for C in range(N)
            if np.any(_block(f, R, C) == 15)}


def containers_box(f):
    """Real containers = 9-bordered/2-interior boxes (fsjjayjoeg).  Solid all-2
    blocks (decoys) have no 9 and are excluded."""
    border = set()
    for R in range(N):
        for C in range(N):
            s = set(_block(f, R, C).flatten().tolist())
            if 9 in s and 2 in s and not ({4, 14, 12, 5, 15} & s):
                border.add((R, C))
    allc = set(border); q = collections.deque(border)
    while q:
        cur = q.popleft()
        for dr, dc in DIRS.values():
            nx = (cur[0] + dr, cur[1] + dc)
            if (0 <= nx[0] < N and 0 <= nx[1] < N and nx not in allc
                    and set(_block(f, nx[0], nx[1]).flatten().tolist()) <= {2}):
                allc.add(nx); q.append(nx)
    return allc


def cont_bbox(f):
    cs = containers_box(f)
    if not cs:
        return set(), set()
    r0 = min(r for r, _ in cs); r1 = max(r for r, _ in cs)
    c0 = min(c for _, c in cs); c1 = max(c for _, c in cs)
    return cs, {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}


def _bfs_carry2(start, goal, off, avb, carb):
    odr, odc = off

    def ok(a):
        if not (0 <= a[0] < N and 0 <= a[1] < N):
            return False
        c = (a[0] + odr, a[1] + odc)
        if not (0 <= c[0] < N and 0 <= c[1] < N):
            return False
        return a not in avb and c not in carb

    if start == goal:
        return []
    q = collections.deque([start]); prev = {start: (None, None)}
    while q:
        cur = q.popleft()
        for a, (dr, dc) in DIRS.items():
            nx = (cur[0] + dr, cur[1] + dc)
            if nx in prev or not ok(nx):
                continue
            prev[nx] = (cur, a)
            if nx == goal:
                acts = []; nd = nx
                while prev[nd][0] is not None:
                    p, a2 = prev[nd]; acts.append(a2); nd = p
                return acts[::-1]
            q.append(nx)
    return None


def _deliver_count(f):
    cs, bb = cont_bbox(f)
    return sum(1 for c in carriers_all(f) if c in bb)


def _run_clone(e, acts):
    for a in acts:
        if e.terminal():
            return
        e.step(a)


def kill_adversaries(env, step, lvl):
    """Destroy colour-15 sprites (each ACT facing one removes it)."""
    for _ in range(6):
        f = env.frame(); a = avatar_cell(f)
        advs = col15_cells(f)
        if not advs or a is None:
            return
        c5 = colour5_cells(f); tex = wallA_cells(f); cars = carriers_all(f)
        best = None
        for adv in advs:
            for delta, face, off in SIDES:
                appr = (adv[0] + delta[0], adv[1] + delta[1])
                if not (0 <= appr[0] < N and 0 <= appr[1] < N):
                    continue
                if appr in c5 or appr in tex or appr in cars or appr in advs:
                    continue
                nav = _bfs_free(a, {appr}, c5 | tex | cars | (advs - {adv}))
                if nav is None:
                    continue
                acts = nav + [face, ACT]
                if best is None or len(acts) < len(best):
                    best = acts
        if best is None:
            return
        for act in best:
            step(act)
            if env.terminal() or env.levels_completed != lvl:
                return


def solo_deliver(env, step, lvl):
    """Carry every loose carrier to the real container; verify each plan on a
    clone (defeats desync from the moving adversary)."""
    stall = 0
    while not env.terminal() and env.levels_completed == lvl and stall < 2:
        f = env.frame(); a = avatar_cell(f)
        if a is None:
            return
        cs, bb = cont_bbox(f); cars = carriers_all(f)
        loose = [c for c in cars if c not in bb]
        targets = [c for c in cs if c not in cars]
        if not loose or not targets:
            return
        c5 = colour5_cells(f); tex = wallA_cells(f); advs = col15_cells(f)
        base = _deliver_count(f); committed = False
        loose.sort(key=lambda c: abs(c[0] - a[0]) + abs(c[1] - a[1]))
        for carrier in loose:
            avb = c5 | tex | advs | (cars - {carrier})
            carb = c5 | advs | (cars - {carrier})
            cand = []
            for tgt in targets:
                for delta, face, off in SIDES:
                    appr = (carrier[0] + delta[0], carrier[1] + delta[1])
                    if not (0 <= appr[0] < N and 0 <= appr[1] < N):
                        continue
                    if appr in avb or appr in cars:
                        continue
                    nav = _bfs_free(a, {appr}, avb | cars)
                    if nav is None:
                        continue
                    goal_av = (tgt[0] - off[0], tgt[1] - off[1])
                    if not (0 <= goal_av[0] < N and 0 <= goal_av[1] < N):
                        continue
                    carry = _bfs_carry2(appr, goal_av, off, avb, carb)
                    if carry is None:
                        continue
                    cand.append((len(nav) + len(carry),
                                 nav + [face, ACT] + carry + [ACT, AWAY[face]]))
            cand.sort(key=lambda t: t[0])
            for _, acts in cand[:8]:
                cl = env.clone(); _run_clone(cl, acts)
                if not cl.terminal() and (_deliver_count(cl.frame()) > base
                                          or cl.levels_completed > lvl):
                    for act in acts:
                        step(act)
                        if env.terminal() or env.levels_completed != lvl:
                            return
                    committed = True; break
            if committed:
                break
        if not committed:
            stall += 1
        else:
            stall = 0


def solo_controller(env, step, lvl):
    kill_adversaries(env, step, lvl)
    solo_deliver(env, step, lvl)


def _idle(env, step, av, f):
    """Take a move that is blocked (bumps a wall) so the avatar stays put and
    ticks the helpers without accidentally attaching anything."""
    blk = wallA_cells(f) | colour5_cells(f)
    for d, (dr, dc) in DIRS.items():
        nx = (av[0] + dr, av[1] + dc)
        if nx in blk or not (0 <= nx[0] < N and 0 <= nx[1] < N):
            step(d)
            return
    step(UP)


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

        # ---- TRAPPED-AVATAR EJECTION (L4+): if the avatar is sealed in a room
        # of textured colour-2 walls with no container reachable, switch to the
        # eject-to-helper controller for this whole level. ----
        av0 = avatar_cell(f)
        if lvl >= 3 and av0 is not None and wallA_cells(f):
            reg0 = avatar_region(f, av0)
            conts0 = cont2_cells(f)
            # trapped == small enclosed region that reaches no container
            if len(reg0) <= 40 and not (reg0 & conts0):
                inside0 = [c for c in carriers_all(f) if c not in conts0 and any(
                    (c[0] + dr, c[1] + dc) in reg0 for dr, dc in DIRS.values())]
                if inside0:
                    eject_controller(env, step, lvl)
                    continue

        # ---- SOLO delivery (L6+): no helper on the board -> the avatar must
        # destroy any colour-15 adversary and carry the carriers itself. ----
        if lvl >= 5 and not helper_cells(f) and av0 is not None:
            before = env.levels_completed
            solo_controller(env, step, lvl)
            if env.terminal() or env.levels_completed != before:
                continue
            # solo made no progress -> fall through / avoid spinning
            stuck += 1
            if stuck > 3:
                break
            continue

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

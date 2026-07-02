import numpy as np
import collections

# Action codes (discovered by experiment):
#  1=up  2=down  3=left  4=right  5=interact(attach/release)
UP, DOWN, LEFT, RIGHT, ACT = 1, 2, 3, 4, 5
DIRS = {UP: (-1, 0), DOWN: (1, 0), LEFT: (0, -1), RIGHT: (0, 1)}
N = 16  # macro grid is 16x16 (each macro cell = 4x4 pixels)


def macro_center(f, R, C):
    """The 2x2 center colours of macro cell (R,C)."""
    return f[4 * R + 1:4 * R + 3, 4 * C + 1:4 * C + 3]


def avatar_cell(f):
    ys, xs = np.where(f == 14)
    if len(ys) == 0:
        return None
    return (int(ys.min()) // 4, int(xs.min()) // 4)


def scan(f):
    """Return (carrier_cells, walls)."""
    carriers = set()
    walls = set()
    for R in range(N):
        for C in range(N):
            cen = macro_center(f, R, C)
            if np.all(cen == 9):
                carriers.add((R, C))
            blk = f[4 * R:4 * R + 4, 4 * C:4 * C + 4]
            if np.any(blk == 7):
                walls.add((R, C))
    return carriers, walls


def container_cells(f):
    """All macro cells of the container = bbox of macro cells whose center is the
    container colour (2). Includes both empty (center 2) and already-filled
    (center 9) slots, so we never try to move an already-delivered carrier."""
    twos = [(R, C) for R in range(N) for C in range(N)
            if np.all(macro_center(f, R, C) == 2)]
    if not twos:
        return set()
    r0 = min(r for r, _ in twos); r1 = max(r for r, _ in twos)
    c0 = min(c for _, c in twos); c1 = max(c for _, c in twos)
    return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}


def agent_cells(f):
    """Macro cells occupied by the autonomous colour-12 agent (helper/adversary)."""
    out = set()
    for R in range(N):
        for C in range(N):
            if np.any(f[4 * R:4 * R + 4, 4 * C:4 * C + 4] == 12):
                out.add((R, C))
    return out


def bfs(start, goal, free_fn):
    if start == goal:
        return []
    q = collections.deque([start])
    prev = {start: (None, None)}
    while q:
        cur = q.popleft()
        for a, (dr, dc) in DIRS.items():
            nxt = (cur[0] + dr, cur[1] + dc)
            if nxt in prev:
                continue
            if not (0 <= nxt[0] < N and 0 <= nxt[1] < N):
                continue
            if not free_fn(nxt):
                continue
            prev[nxt] = (cur, a)
            if nxt == goal:
                acts = []
                node = nxt
                while prev[node][0] is not None:
                    p, a2 = prev[node]
                    acts.append(a2)
                    node = p
                return acts[::-1]
            q.append(nxt)
    return None


def bfs_carry(start, goal, offset, free_fn):
    """Carry BFS: both avatar cell AND carrier cell (avatar+offset) must be free."""
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
    q = collections.deque([start])
    prev = {start: (None, None)}
    while q:
        cur = q.popleft()
        for a, (dr, dc) in DIRS.items():
            nxt = (cur[0] + dr, cur[1] + dc)
            if nxt in prev or not unit_ok(nxt):
                continue
            prev[nxt] = (cur, a)
            if nxt == goal:
                acts = []
                node = nxt
                while prev[node][0] is not None:
                    p, a2 = prev[node]
                    acts.append(a2)
                    node = p
                return acts[::-1]
            q.append(nxt)
    return None


# (attach_cell_delta, face_action, carrier_offset_from_avatar)
SIDES = [
    ((1, 0), UP, (-1, 0)),     # attach from below, carrier above
    ((-1, 0), DOWN, (1, 0)),   # from above, carrier below
    ((0, -1), RIGHT, (0, 1)),  # from left, carrier right
    ((0, 1), LEFT, (0, -1)),   # from right, carrier left
]


def solve(env):
    max_real = 560
    real_used = [0]

    def step(a):
        if env.terminal():
            return None
        real_used[0] += 1
        return env.step(a)

    stuck = 0
    cur_level = -1
    slotset = set()
    while not env.terminal() and real_used[0] < max_real:
        f = env.frame()
        lvl = env.levels_completed
        if lvl != cur_level:
            slotset = container_cells(f)
            cur_level = lvl
            stuck = 0
        if not slotset:
            slotset = container_cells(f)
            if not slotset:
                break

        carriers, walls = scan(f)
        agents = agent_cells(f)
        empty = [s for s in slotset if not np.all(macro_center(f, s[0], s[1]) == 9)]
        movable = [c for c in carriers if c not in slotset]
        if not empty or not movable:
            break

        av = avatar_cell(f)
        if av is None:
            break

        carrier_cells = carriers | agents

        def make_plan():
            cand = []
            for car in movable:
                for slot in empty:
                    d = abs(car[0] - slot[0]) + abs(car[1] - slot[1])
                    cand.append((d, car, slot))
            cand.sort(key=lambda x: x[0])
            for _, car, slot in cand:
                for delta, face, off in SIDES:
                    acell = (car[0] + delta[0], car[1] + delta[1])
                    if not (0 <= acell[0] < N and 0 <= acell[1] < N):
                        continue

                    def nav_free(cell):
                        if cell in walls or cell in carrier_cells or cell in slotset:
                            return False
                        return True

                    if not nav_free(acell):
                        continue
                    nav = bfs(av, acell, nav_free)
                    if nav is None:
                        continue

                    def carry_free(cell, _car=car):
                        if cell in walls:
                            return False
                        if cell in carrier_cells and cell != _car:
                            return False
                        return True

                    goal_av = (slot[0] - off[0], slot[1] - off[1])
                    if not (0 <= goal_av[0] < N and 0 <= goal_av[1] < N):
                        continue
                    carry = bfs_carry(acell, goal_av, off, carry_free)
                    if carry is None:
                        continue
                    return nav, face, carry, car, slot
            return None

        plan = make_plan()
        if plan is None:
            stuck += 1
            if stuck > 2:
                break
            step(RIGHT)
            continue
        stuck = 0
        nav, face, carry, car, slot = plan

        for a in nav:
            step(a)
            if env.terminal():
                return
        step(face)   # orient toward carrier (blocked move just turns)
        step(ACT)     # attach
        for a in carry:
            step(a)
            if env.terminal():
                return
        step(ACT)     # release into slot

        f2 = env.frame()
        if not np.all(macro_center(f2, slot[0], slot[1]) == 9):
            stuck += 1
            if stuck > 3:
                break

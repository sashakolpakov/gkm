import numpy as np
import collections

# wa30: 16x16 macro grid, each cell = 4x4 pixels, total 64x64 frame
N = 16
UP, DOWN, LEFT, RIGHT, ACT = 1, 2, 3, 4, 5
DIRS = {UP: (-1, 0), DOWN: (1, 0), LEFT: (0, -1), RIGHT: (0, 1)}

# (attach_cell_delta_from_carrier, face_action, carrier_offset_from_avatar)
ATTACH_SIDES = [
    ((1, 0),  UP,    (-1, 0)),
    ((-1, 0), DOWN,  (1, 0)),
    ((0, -1), RIGHT, (0, 1)),
    ((0, 1),  LEFT,  (0, -1)),
]


def macro_center(f, R, C):
    return f[4*R+1:4*R+3, 4*C+1:4*C+3]


def find_avatar(f, color=14):
    ys, xs = np.where(f == color)
    if len(ys) == 0:
        return None
    return (int(ys.min()) // 4, int(xs.min()) // 4)


def find_color_macro(f, color):
    return [(R, C) for R in range(N) for C in range(N)
            if np.all(macro_center(f, R, C) == color)]


def find_any_macro(f, color):
    return [(R, C) for R in range(N) for C in range(N)
            if np.any(f[4*R:4*R+4, 4*C:4*C+4] == color)]


def bfs_grid(start, goal, free_fn):
    if start == goal:
        return []
    q = collections.deque([start])
    prev = {start: (None, None)}
    while q:
        cur = q.popleft()
        for a, (dr, dc) in DIRS.items():
            nxt = (cur[0]+dr, cur[1]+dc)
            if nxt in prev:
                continue
            if not (0 <= nxt[0] < N and 0 <= nxt[1] < N):
                continue
            if not free_fn(nxt):
                continue
            prev[nxt] = (cur, a)
            if nxt == goal:
                acts, node = [], nxt
                while prev[node][0] is not None:
                    p, a2 = prev[node]
                    acts.append(a2)
                    node = p
                return acts[::-1]
            q.append(nxt)
    return None


def bfs_carry_grid(start, goal, offset, free_fn):
    odr, odc = offset

    def unit_ok(av):
        if not (0 <= av[0] < N and 0 <= av[1] < N):
            return False
        car = (av[0]+odr, av[1]+odc)
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
            nxt = (cur[0]+dr, cur[1]+dc)
            if nxt in prev or not unit_ok(nxt):
                continue
            prev[nxt] = (cur, a)
            if nxt == goal:
                acts, node = [], nxt
                while prev[node][0] is not None:
                    p, a2 = prev[node]
                    acts.append(a2)
                    node = p
                return acts[::-1]
            q.append(nxt)
    return None


def make_step_budget(env, max_real):
    """Return (step_fn, budget_ok_fn): step_fn guards terminal and counts; budget_ok_fn checks remaining."""
    used = [0]

    def step(a):
        if env.terminal():
            return None
        used[0] += 1
        return env.step(a)

    def budget_ok():
        return used[0] < max_real

    return step, budget_ok


def sokoban_deliver(env, carrier_color=9, slot_color=2, wall_color=7, agent_color=12, max_real=560):
    """Deliver all loose carriers into container slots via attach/carry/release."""
    step, budget_ok = make_step_budget(env, max_real)

    stuck = 0
    cur_level = -1
    slotset = set()
    while not env.terminal() and budget_ok():
        f = env.frame()
        lvl = env.levels_completed
        if lvl != cur_level:
            slotset = set(find_color_macro(f, slot_color))
            cur_level = lvl
            stuck = 0
        if not slotset:
            slotset = set(find_color_macro(f, slot_color))
            if not slotset:
                break

        walls = set(find_any_macro(f, wall_color))
        agents = set(find_any_macro(f, agent_color))
        carriers = set(find_color_macro(f, carrier_color))
        empty = [s for s in slotset if not np.all(macro_center(f, s[0], s[1]) == carrier_color)]
        movable = [c for c in carriers if c not in slotset]
        if not empty or not movable:
            break

        av = find_avatar(f)
        if av is None:
            break

        carrier_cells = carriers | agents

        def make_plan():
            cand = sorted(
                [(abs(c[0]-s[0]) + abs(c[1]-s[1]), c, s) for c in movable for s in empty]
            )
            for _, car, slot in cand:
                for delta, face, off in ATTACH_SIDES:
                    acell = (car[0]+delta[0], car[1]+delta[1])
                    if not (0 <= acell[0] < N and 0 <= acell[1] < N):
                        continue

                    def nav_free(cell, _w=walls, _cc=carrier_cells, _ss=slotset):
                        return cell not in _w and cell not in _cc and cell not in _ss

                    if not nav_free(acell):
                        continue
                    nav = bfs_grid(av, acell, nav_free)
                    if nav is None:
                        continue

                    def carry_free(cell, _w=walls, _cc=carrier_cells, _car=car):
                        return cell not in _w and (cell not in _cc or cell == _car)

                    goal_av = (slot[0]-off[0], slot[1]-off[1])
                    if not (0 <= goal_av[0] < N and 0 <= goal_av[1] < N):
                        continue
                    carry = bfs_carry_grid(acell, goal_av, off, carry_free)
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
        step(face)
        step(ACT)
        for a in carry:
            step(a)
            if env.terminal():
                return
        step(ACT)

        f2 = env.frame()
        if not np.all(macro_center(f2, slot[0], slot[1]) == carrier_color):
            stuck += 1
            if stuck > 3:
                break

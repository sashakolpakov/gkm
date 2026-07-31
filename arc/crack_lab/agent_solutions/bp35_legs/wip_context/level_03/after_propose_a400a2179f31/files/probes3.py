"""Probe bodies for L3. Registered into p3.PROBES."""
from collections import Counter

from p3 import colours, probe, show, tup
from legs import (AVATAR_ROW, COL_ANCHORS, GRID_COLS, GRID_ROWS, ROW_ANCHORS,
                  avatar_column, band_grid, band_shift, click_action,
                  moves_used)


@probe
def start(env):
    """L3 opening frame: symbolic grid, palette, and env.actions."""
    print('actions =', getattr(env, 'actions', None))
    show(env, 'L3 start')
    print('colours =', colours(env))


@probe
def raw(env, r0='0', r1='64'):
    """Raw pixel rows, compactly, to check the lattice assumption on L3."""
    f = env.frame()
    for r in range(int(r0), min(int(r1), 64)):
        print(f'{r:2d} ' + ''.join(f'{int(v):x}' for v in f[r]))


@probe
def keys(env):
    """What does each key action do on a clone at L3 start?"""
    base = env.frame()
    for a in (3, 4, 7):
        c = env.clone()
        c.step(a)
        f = c.frame()
        diff = sum(1 for r in range(64) for x in range(64)
                   if int(base[r][x]) != int(f[r][x]))
        print(f'action {a}: diff={diff} col={avatar_column(f)} '
              f'shift={band_shift(base, f)} moves={moves_used(f)} '
              f'lvl={c.levels_completed} term={c.terminal()}')
        show(c, f'after {a}')


@probe
def seven(env, n='6'):
    """Repeat action 7 -- is it a toggle, a fire, a no-op, a drop?"""
    c = env.clone()
    prev = c.frame()
    show(c, 'before any 7')
    for k in range(int(n)):
        c.step(7)
        f = c.frame()
        diff = sum(1 for r in range(64) for x in range(64)
                   if int(prev[r][x]) != int(f[r][x]))
        print(f'7 x{k+1}: diff={diff} col={avatar_column(f)} '
              f'shift={band_shift(prev, f)} lvl={c.levels_completed} '
              f'term={c.terminal()} colours={colours(c)}')
        show(c, f'after 7 x{k+1}')
        prev = f


@probe
def idle(env, action='4', n='30'):
    """Burn moves to measure the rising floor on L3."""
    c = env.clone()
    a = int(action)
    for k in range(int(n)):
        c.step(a)
        f = c.frame()
        print(f'{k+1:2d} col={avatar_column(f)} moves={moves_used(f)} '
              f'lvl={c.levels_completed} term={c.terminal()}')
        if avatar_column(f) is None or c.terminal():
            show(c, 'death frame')
            break


@probe
def blobs(env, colors='12,14,9,11,7,15'):
    """Connected components of the interesting colours, with shape info."""
    import perception as P
    want = [int(x) for x in colors.split(',')]
    f = env.frame()
    for b in P.connected_components(f, colors=want, min_area=1):
        r0, c0, r1, c1 = b.bbox
        band = f'rows {r0//6}-{r1//6} cols {(c0-13)//6}-{(c1-13)//6}'
        print(f'colour {b.color:2d} bbox={b.bbox} size={b.size} area={b.area}  bands: {band}')


@probe
def walk(env, action='4', n='4'):
    """Take one key action repeatedly, showing the symbolic grid each time."""
    c = env.clone()
    a = int(action)
    prev = c.frame()
    show(c, 'before')
    for k in range(int(n)):
        c.step(a)
        f = c.frame()
        print(f'>> {a} #{k+1}: col={avatar_column(f)} shift={band_shift(prev, f)} '
              f'lvl={c.levels_completed} term={c.terminal()} c12={colours(c).get(12)}')
        show(c, f'after {a} x{k+1}')
        prev = f


@probe
def clicks(env):
    """Click every band cell on a fresh clone; report symbol -> effect."""
    base = env.frame()
    g0 = band_grid(base)
    c0 = colours(env)
    for i in range(GRID_ROWS):
        for j in range(GRID_COLS):
            c = env.clone()
            c.step(*click_action(i, j))
            f = c.frame()
            diff = sum(1 for r in range(63) for x in range(64)
                       if int(base[r][x]) != int(f[r][x]))
            if diff == 0:
                continue
            cn = colours(c)
            dc = {k: cn.get(k, 0) - c0.get(k, 0) for k in set(cn) | set(c0)
                  if cn.get(k, 0) != c0.get(k, 0)}
            print(f'({i},{j}) sym={g0[i][j]!r} diff={diff:4d} '
                  f'shift={band_shift(base, f)} col={avatar_column(f)} '
                  f'lvl={c.levels_completed} term={c.terminal()} dcol={dc}')


def _parse(tok):
    if tok.startswith('c'):
        i, j = tok[2:].split(',')
        return click_action(int(i), int(j))
    return (int(tok),)


@probe
def seq(env, *toks):
    """Run a hand-written action sequence: '4', '3', or 'c:i,j' for a click."""
    c = env.clone()
    prev = c.frame()
    total = 0
    show(c, 'start')
    for t in toks:
        a = _parse(t)
        c.step(*a)
        f = c.frame()
        s = band_shift(prev, f)
        total += s
        print(f'>> {t} -> col={avatar_column(f)} shift={s} total={total} '
              f'moves={moves_used(f)} lvl={c.levels_completed} term={c.terminal()}')
        show(c, t)
        prev = f
    return c


def cell_area(f, i, j):
    """(centre colour, count of that colour inside the 6x6 band cell)."""
    r0, c0 = 6 * i, 13 + 6 * j
    ctr = int(f[ROW_ANCHORS[i]][COL_ANCHORS[j]])
    n = sum(1 for r in range(r0, r0 + 6) for c in range(c0, c0 + 6)
            if 0 <= r < 64 and 0 <= c < 64 and int(f[r][c]) == ctr)
    return ctr, n


@probe
def cells(env, *toks):
    """Run a sequence, then table every cell as (centre colour, fill area)."""
    c = env.clone()
    for t in toks:
        c.step(*_parse(t))
    f = c.frame()
    print(f'lvl={c.levels_completed} term={c.terminal()} moves={moves_used(f)} '
          f'col={avatar_column(f)}')
    for i in range(GRID_ROWS):
        cs = []
        for j in range(GRID_COLS):
            ctr, n = cell_area(f, i, j)
            cs.append(f'{ctr:2d}/{n:2d}')
        mark = '<' if i == AVATAR_ROW else ' '
        print(f'  {i} ' + ' '.join(cs) + mark)


@probe
def prizecell(env):
    """Replay the L1/L2 prefix and table the frame just before each level fires."""
    import json
    path = json.load(open('checkpoint.json'))['final_path']
    c = None
    import gkm_arena as A  # noqa

    def run(prefix):
        e = FRESH.clone()
        for a in prefix:
            e.step(*tup(a))
        return e
    lvl = 0
    for k in range(len(path)):
        e = run(path[:k + 1])
        if e.levels_completed > lvl:
            before = run(path[:k])
            f = before.frame()
            print(f'--- level {lvl+1} fires on action {path[k]} (index {k})')
            for i in range(GRID_ROWS):
                print('  %d ' % i + ' '.join(
                    '%2d/%2d' % cell_area(f, i, j) for j in range(GRID_COLS)))
            lvl = e.levels_completed


@probe
def search(env, expansions='800'):
    """Try the existing search leg on L3 and report what it finds."""
    import time
    from legs import climb_search, run_actions
    c = env.clone()
    t = time.time()
    acts = climb_search(c, max_expansions=int(expansions))
    print(f'search took {time.time()-t:.1f}s -> {len(acts)} actions')
    print('actions =', acts)
    d = env.clone()
    run_actions(d, acts)
    f = d.frame()
    print(f'result lvl={d.levels_completed} term={d.terminal()} '
          f'moves={moves_used(f)} col={avatar_column(f)}')
    show(d, 'end of plan')

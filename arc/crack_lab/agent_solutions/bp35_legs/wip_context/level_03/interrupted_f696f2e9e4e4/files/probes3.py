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

import json, sys, traceback
from collections import Counter
sys.path.insert(0, '/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
from perception import arr

PATH = json.load(open('checkpoint.json'))['final_path']
UP, DOWN, LEFT, RIGHT, USE = 1, 2, 3, 4, 5
NAME = {1: 'U', 2: 'D', 3: 'L', 4: 'R', 5: 'S'}
MARK9 = {(6, 12), (9, 9), (9, 30), (27, 12)}


def cross(e):
    """(centre, up, down, left, right, ncells) of the colour-9 cross."""
    f = arr(e.frame())
    pts = {(int(y), int(x)) for y, x in zip(*(f == 9).nonzero())} - MARK9
    z = [(int(y), int(x)) for y, x in zip(*(f == 0).nonzero())]
    pts |= set(z)  # selection dot punches a hole in the shape
    if not pts:
        return None
    rowc = Counter(p[0] for p in pts)
    colc = Counter(p[1] for p in pts)
    cr = rowc.most_common(1)[0][0]
    cc = colc.most_common(1)[0][0]
    vert = sorted(p[0] for p in pts if p[1] == cc)
    horz = sorted(p[1] for p in pts if p[0] == cr)
    u, d = cr - vert[0], vert[-1] - cr
    l, r = cc - horz[0], horz[-1] - cc
    ideal = {(cr + k, cc) for k in range(-u, d + 1)} | {(cr, cc + k) for k in range(-l, r + 1)}
    return (cr, cc, u, d, l, r, len(pts), u + d + l + r, pts == ideal)


def show(tag, e):
    c = cross(e)
    if c is None:
        print(f'  {tag} gone'); return
    cr, cc, u, d, l, r, n, s, perfect = c
    print(f'  {tag} ctr=({cr},{cc}) u={u} d={d} l={l} r={r} sum={s} n={n} perfect={perfect}')


def run(env, plan, tag, pre=()):
    c = env.clone()
    for a in pre:
        c.step(a)
    print(f'--- {tag} ---')
    show('init', c)
    for a in plan:
        try:
            c.step(a)
        except Exception:
            print('  DIED'); return
        show(NAME[a], c)
    return c


def prog(env):
  try:
    for a in PATH:
        env.step(a)
    run(env, [UP]*4, 'plus UP x4 (toward top edge)', pre=[USE])
    run(env, [RIGHT]*3, 'plus RIGHT x3 (toward right edge)', pre=[USE])
    run(env, [LEFT]*6, 'plus LEFT x6 (open, toward fixture col)', pre=[USE])
    # put the cross arm inside the fixture band, then push left
    run(env, [LEFT]*8, 'plus DOWN x5 then LEFT x8 (into fixture)',
        pre=[USE, DOWN, DOWN, DOWN, DOWN, DOWN])
  except Exception:
    traceback.print_exc()


A.run_program('re86', prog)

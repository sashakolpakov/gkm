import json, sys, traceback
sys.path.insert(0, '/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
from perception import arr

PATH = json.load(open('checkpoint.json'))['final_path']
UP, DOWN, LEFT, RIGHT, USE = 1, 2, 3, 4, 5
NAME = {1: 'U', 2: 'D', 3: 'L', 4: 'R', 5: 'S'}
MARK11 = {(30, 45), (30, 54), (57, 45), (57, 54)}


def cells11(e):
    f = arr(e.frame())
    pts = {(int(y), int(x)) for y, x in zip(*(f == 11).nonzero())}
    return pts - MARK11


def describe(pts):
    if not pts:
        return 'empty'
    rs = [p[0] for p in pts]; cs = [p[1] for p in pts]
    r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
    h, w = r1 - r0 + 1, c1 - c0 + 1
    rect = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)
            if r in (r0, r1) or c in (c0, c1)}
    return (f'bbox=({r0},{c0},{r1},{c1}) h={h} w={w} h+w={h+w} n={len(pts)} '
            f'perfect_outline={pts == rect} missing={len(rect - pts)} extra={len(pts - rect)}')


def prog(env):
  try:
    for a in PATH:
        env.step(a)
    c = env.clone()
    print('init ', describe(cells11(c)))
    for i, a in enumerate([UP]*3 + [RIGHT]*9):
        c.step(a)
        print(f'{i:2d} {NAME[a]} ', describe(cells11(c)))
    print('\n=== pixel dump rows 18-58, cols 8-40 (final) ===')
    f = arr(c.frame())
    print('    ' + ''.join(str(x % 10) for x in range(8, 41)))
    for r in range(18, 59):
        print(f'{r:3d} ' + ''.join('.' if int(v) == 5 else format(int(v), 'x')
                                   for v in f[r][8:41]))
  except Exception:
    traceback.print_exc()


A.run_program('re86', prog)

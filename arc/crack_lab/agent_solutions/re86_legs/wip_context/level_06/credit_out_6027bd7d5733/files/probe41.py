import json, sys, traceback
sys.path.insert(0, '/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
from perception import arr

PATH = json.load(open('checkpoint.json'))['final_path']
UP, DOWN, LEFT, RIGHT, USE = 1, 2, 3, 4, 5
NAME = {1: 'U', 2: 'D', 3: 'L', 4: 'R', 5: 'S'}
MARK11 = [(30, 45), (30, 54), (57, 45), (57, 54)]
MARK9 = [(6, 12), (9, 9), (9, 30), (27, 12)]


def cells(e, color, exclude):
    f = arr(e.frame())
    pts = {(int(y), int(x)) for y, x in zip(*(f == color).nonzero())}
    return pts - set(exclude)


def bbox(pts):
    if not pts:
        return None
    rs = [p[0] for p in pts]; cs = [p[1] for p in pts]
    return (min(rs), min(cs), max(rs), max(cs))


def cov(e, color, marks):
    f = arr(e.frame())
    return [m for m in marks if int(f[m[0], m[1]]) == color and _is_shape(f, m, color)]


def _is_shape(f, m, color):
    return True


def prog(env):
  try:
    for a in PATH:
        env.step(a)
    plan = [UP]*3 + [RIGHT]*4 + [UP]*8 + [RIGHT]*9 + [DOWN]*10
    c = env.clone()
    for i, a in enumerate(plan):
        c.step(a)
        pts = cells(c, 11, MARK11)
        bb = bbox(pts)
        h = bb[2]-bb[0]+1; w = bb[3]-bb[1]+1
        print(f'{i:2d} {NAME[a]} bbox={bb} h={h} w={w} h+w={h+w} n={len(pts)} '
              f'lvl={c.levels_completed}')
    print('\n=== frame around the 11 markers ===')
    f = arr(c.frame())
    print('     ' + ''.join(str(x % 10) for x in range(42, 58)))
    for r in range(28, 60):
        print(f'{r:3d}  ' + ''.join('.' if int(v) == 5 else format(int(v), 'x')
                                    for v in f[r][42:58]))
    print('marker cells now:', [(m, int(f[m[0], m[1]])) for m in MARK11])
    print('lvl', c.levels_completed, 'hud', int((f[63] == 15).sum()))
    print('ring4 count', int((f == 4).sum()))
  except Exception:
    traceback.print_exc()


A.run_program('re86', prog)

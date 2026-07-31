import json, sys, traceback
sys.path.insert(0, '/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
from perception import arr, connected_components

PATH = json.load(open('checkpoint.json'))['final_path']
UP, DOWN, LEFT, RIGHT, USE = 1, 2, 3, 4, 5


def info(e):
    f = arr(e.frame())
    zeros = [(int(y), int(x)) for y, x in zip(*(f == 0).nonzero())]
    b11 = [b for b in connected_components(f, colors=(11,), min_area=20)]
    bb = b11[0].bbox if b11 else None
    return zeros, bb, int((f == 11).sum()), int((f == 1).sum())


def prog(env):
  try:
    for a in PATH:
        env.step(a)
    print('=== square: 3 UP then RIGHT until stuck ===')
    c = env.clone()
    for i, a in enumerate([UP]*3 + [RIGHT]*10):
        c.step(a)
        z, bb, n11, n1 = info(c)
        print(f'{i:2d} a={a} zeros={z} sq_bbox={bb} n11={n11} n1={n1}')

    print('\n=== dump region rows 26-50 cols 6-40 at the stuck state ===')
    f = arr(c.frame())
    for r in range(26, 51):
        print(f'{r:2d} ' + ''.join('.' if int(v) == 5 else format(int(v), 'x')
                                   for v in f[r][6:41]))

    print('\n=== budget: how many moves until no frame? ===')
    d = env.clone()
    n = 0
    while True:
        try:
            d.step(USE)
            arr(d.frame())
            n += 1
        except Exception as ex:
            print('died after', n, 'USE presses; term', d.terminal(),
                  'lvl', d.levels_completed)
            break
        if n > 400:
            print('no death after', n); break
  except Exception:
    traceback.print_exc()


A.run_program('re86', prog)

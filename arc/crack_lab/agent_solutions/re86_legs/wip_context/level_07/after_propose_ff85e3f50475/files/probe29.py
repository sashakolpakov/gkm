import json, sys, traceback
sys.path.insert(0, '/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
from perception import arr, USE, LEFT, RIGHT, UP, DOWN
from collections import Counter

PATH = json.load(open('checkpoint.json'))['final_path']
NAME = {1: 'U', 2: 'D', 3: 'L', 4: 'R'}


def shape_color(e):
    f = arr(e.frame())
    mv = e.clone(); mv.step(RIGHT); g = arr(mv.frame())
    cnt = Counter(int(f[r, c]) for r, c in zip(*((f != g).nonzero()))
                  if int(g[r, c]) == 5 and int(f[r, c]) not in (0, 5))
    return cnt.most_common(1)[0][0] if cnt else None


def prog(env):
  try:
    for a in PATH:
        env.step(a)
    s = env.clone()
    cur = (42, 24)
    last = shape_color(s)
    print('start', cur, 'color', last)
    seq = [(DOWN, (3, 0))] * 4 + [(LEFT, (0, -3))] * 6 + \
          [(UP, (-3, 0))] * 13 + [(RIGHT, (0, 3))] * 8
    for a, d in seq:
        s.step(a)
        cur = (cur[0] + d[0], cur[1] + d[1])
        col = shape_color(s)
        if col != last:
            print(f'  at {cur} color {last} -> {col}')
            last = col
    print('end', cur, 'color', last)
  except Exception:
    traceback.print_exc()


A.run_program('re86', prog)

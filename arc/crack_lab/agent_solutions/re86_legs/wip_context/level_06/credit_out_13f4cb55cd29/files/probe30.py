import json, sys, traceback
sys.path.insert(0, '/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
from perception import arr, USE, LEFT, RIGHT, UP, DOWN
from collections import Counter

PATH = json.load(open('checkpoint.json'))['final_path']

ROUTES = [
    [(42, 24), (54, 24), (54, 9), (54, 30), (15, 30)],
    [(18, 30), (45, 30), (45, 57), (36, 57), (36, 51)],
    [(33, 54), (54, 54), (54, 21), (51, 21), (51, 33)],
]


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
    n = 0
    for i, route in enumerate(ROUTES):
        if i:
            s.step(USE); n += 1
        cur = route[0]
        for nxt in route[1:]:
            while cur != nxt:
                if cur[0] != nxt[0]:
                    a, d = (DOWN, (3, 0)) if nxt[0] > cur[0] else (UP, (-3, 0))
                else:
                    a, d = (RIGHT, (0, 3)) if nxt[1] > cur[1] else (LEFT, (0, -3))
                s.step(a); n += 1
                cur = (cur[0] + d[0], cur[1] + d[1])
        print(f'shape{i} at {cur} color {shape_color(s)} lvl {s.levels_completed}')
    print('FINAL levels', s.levels_completed, 'moves', n, 'terminal', s.terminal())
  except Exception:
    traceback.print_exc()


A.run_program('re86', prog)

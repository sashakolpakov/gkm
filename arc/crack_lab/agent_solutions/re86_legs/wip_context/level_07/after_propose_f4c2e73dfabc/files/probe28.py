import json, sys, traceback
sys.path.insert(0, '/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
from perception import arr, USE, LEFT, RIGHT, UP, DOWN
from collections import Counter

PATH = json.load(open('checkpoint.json'))['final_path']
PLAN = [((42, 24), (54, 6), (15, 30)),
        ((18, 30), (54, 57), (36, 51)),
        ((33, 54), (54, 6), (51, 33))]
MARKS = {(27, 51): 8, (33, 57): 8, (36, 42): 8, (42, 54): 8,
         (6, 21): 9, (6, 39): 9, (45, 33): 9, (51, 24): 9,
         (51, 45): 9, (60, 33): 9}


def run(e, cur, dst):
    while cur != dst:
        if cur[0] != dst[0]:
            a, d = (DOWN, (3, 0)) if dst[0] > cur[0] else (UP, (-3, 0))
        else:
            a, d = (RIGHT, (0, 3)) if dst[1] > cur[1] else (LEFT, (0, -3))
        e.step(a)
        cur = (cur[0] + d[0], cur[1] + d[1])
    return cur


def prog(env):
  try:
    for a in PATH:
        env.step(a)
    s = env.clone()
    for i, (start, paint, dst) in enumerate(PLAN):
        if i:
            s.step(USE)
        cur = run(s, start, paint)
        cur = run(s, cur, dst)
    f = arr(s.frame())
    print('colors', dict(Counter(int(v) for v in f.flat)))
    for p, c in sorted(MARKS.items()):
        print(f'marker {p} want {c} now {int(f[p])}')
    print('HUD row63', ''.join(format(int(v), 'x') for v in f[63]))
    print('rows around (27,51):')
    for r in range(24, 46, 1):
        print(f'{r:3d} ' + ''.join('.' if f[r, c] == 5 else format(int(f[r, c]), 'x')
                                   for c in range(38, 64)))
  except Exception:
    traceback.print_exc()


A.run_program('re86', prog)

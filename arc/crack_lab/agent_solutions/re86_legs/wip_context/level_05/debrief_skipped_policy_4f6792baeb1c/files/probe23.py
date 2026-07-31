import json, sys
sys.path.insert(0, '/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
from perception import arr, USE, LEFT, RIGHT, UP, DOWN

PATH = json.load(open('checkpoint.json'))['final_path']
STATIONS = [(3, 3), (3, 54), (27, 3), (52, 3), (52, 54)]


def in_station(r, c):
    return any(r0 <= r <= r0 + 5 and c0 <= c <= c0 + 5 for r0, c0 in STATIONS)


def prog(env):
    for a in PATH:
        env.step(a)
    for i, moves in enumerate([[], [], [LEFT, LEFT, LEFT]]):
        s = env.clone()
        for _ in range(i):
            s.step(USE)
        for m in moves:
            s.step(m)
        f = arr(s.frame())
        ctr = tuple(int(v) for v in list(zip(*((f == 0).nonzero())))[0])
        # shape color: color that vanishes on a RIGHT move
        mv = s.clone(); mv.step(RIGHT); g = arr(mv.frame())
        from collections import Counter
        cnt = Counter(int(f[r, c]) for r, c in zip(*((f != g).nonzero()))
                      if int(g[r, c]) == 5 and int(f[r, c]) not in (0, 5))
        col = cnt.most_common(1)[0][0]
        offs = sorted((int(r) - ctr[0], int(c) - ctr[1])
                      for r, c in zip(*((f == col).nonzero())) if not in_station(r, c))
        dr = [o[0] for o in offs]; dc = [o[1] for o in offs]
        print(f'shape{i} color={col} center={ctr} n={len(offs)} '
              f'dr=[{min(dr)},{max(dr)}] dc=[{min(dc)},{max(dc)}] '
              f'diag={sum(abs(a)==abs(b) for a,b in offs)} '
              f'axis={sum(a==0 or b==0 for a,b in offs)} '
              f'L1max={max(abs(a)+abs(b) for a,b in offs)} '
              f'L1all={len({abs(a)+abs(b) for a,b in offs})}')


A.run_program('re86', prog)

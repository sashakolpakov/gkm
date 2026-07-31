import json, sys
from collections import Counter
sys.path.insert(0, '/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
from perception import arr

PATH = json.load(open('checkpoint.json'))['final_path']
UP, DOWN, LEFT, RIGHT, USE = 1, 2, 3, 4, 5
CROSS = ([USE] + [DOWN]*5 + [LEFT]*3 + [DOWN]*6 + [LEFT]*3
         + [UP]*3 + [RIGHT]*4 + [UP]*10 + [LEFT]*10)
MARK9 = {(6,12), (9,9), (9,30), (27,12)}


def info(e):
    f = arr(e.frame())
    zeros = [(int(r), int(c)) for r, c in zip(*((f == 0).nonzero()))]
    pts = {(int(r), int(c)) for r, c in zip(*((f == 9).nonzero()))} - MARK9
    pts |= {(int(r), int(c)) for r, c in zip(*((f == 0).nonzero()))}
    rows = Counter(r for r, _ in pts); cols = Counter(c for _, c in pts)
    cr = rows.most_common(1)[0][0]; cc = cols.most_common(1)[0][0]
    vr = [r for r, c in pts if c == cc]; hc = [c for r, c in pts if r == cr]
    covered = sum(int(f[p]) == 9 for p in MARK9)
    return (zeros, cr, cc, cr-min(vr), max(vr)-cr, cc-min(hc), max(hc)-cc,
            len(pts), covered)


def prog(env):
    for action in PATH:
        env.step(action)
    c = env.clone()
    for i, action in enumerate(CROSS):
        c.step(action)
        if i in (0, 5, 8, 14, 17, 20, 24, 34, 44):
            print(i, action, info(c), 'lvl', c.levels_completed)
    print('FINAL', info(c), 'lvl', c.levels_completed)


A.run_program('re86', prog)

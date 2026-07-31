import json, sys, traceback
sys.path.insert(0, '/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
from perception import arr, USE, LEFT, RIGHT, UP, DOWN

PATH = json.load(open('checkpoint.json'))['final_path']

PLAN = [((42, 24), (54, 6), (15, 30)),
        ((18, 30), (54, 57), (36, 51)),
        ((33, 54), (54, 6), (51, 33))]


def blacks(e):
    f = arr(e.frame())
    return [(int(r), int(c)) for r, c in zip(*((f == 0).nonzero()))]


def goto(e, cur, dst, tag):
    while cur != dst:
        if cur[0] != dst[0]:
            a, d = (DOWN, (3, 0)) if dst[0] > cur[0] else (UP, (-3, 0))
        else:
            a, d = (RIGHT, (0, 3)) if dst[1] > cur[1] else (LEFT, (0, -3))
        before = arr(e.frame()).copy()
        e.step(a)
        if (before == arr(e.frame())).all():
            print(f'  BLOCKED {tag} at {cur} action {a}')
            return cur
        cur = (cur[0] + d[0], cur[1] + d[1])
        b = blacks(e)
        if len(b) == 1 and b[0] != cur:
            print(f'  DRIFT {tag}: dead-reckon {cur} vs black {b[0]}')
            cur = b[0]
        elif len(b) != 1:
            print(f'  no-unique-black at {cur}: {b}')
    return cur


def prog(env):
    try:
        for a in PATH:
            env.step(a)
        s = env.clone()
        for i, (start, paint, dst) in enumerate(PLAN):
            if i:
                s.step(USE)
            cur = blacks(s)[0]
            print(f'shape{i} center={cur} expect {start}')
            cur = goto(s, cur, paint, f's{i}-paint')
            print('   at paint', cur, 'lvl', s.levels_completed)
            cur = goto(s, cur, dst, f's{i}-dst')
            print('   at dst', cur, 'lvl', s.levels_completed)
        print('FINAL levels', s.levels_completed, 'terminal', s.terminal())
    except Exception:
        traceback.print_exc()


A.run_program('re86', prog)

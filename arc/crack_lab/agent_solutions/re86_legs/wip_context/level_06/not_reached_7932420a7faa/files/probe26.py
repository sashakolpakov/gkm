import json, sys, traceback
sys.path.insert(0, '/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
from perception import arr, USE, LEFT, RIGHT, UP, DOWN
from collections import Counter

PATH = json.load(open('checkpoint.json'))['final_path']


def prog(env):
    try:
        for a in PATH:
            env.step(a)
        s = env.clone()
        for _ in range(4):
            s.step(DOWN)
        for _ in range(6):
            s.step(LEFT)
        f = arr(s.frame())
        print('black pixels', [(int(r), int(c)) for r, c in zip(*((f == 0).nonzero()))])
        print('colors', Counter(int(v) for v in f.flat))
        print('rows 50-58, cols 0-14')
        for r in range(50, 59):
            print(f'{r:3d} ' + ''.join('.' if f[r, c] == 5 else format(int(f[r, c]), 'x')
                                       for c in range(0, 15)))
        # what color is the shape now?
        mv = s.clone(); mv.step(RIGHT); g = arr(mv.frame())
        cnt = Counter(int(f[r, c]) for r, c in zip(*((f != g).nonzero()))
                      if int(g[r, c]) == 5 and int(f[r, c]) not in (0, 5))
        print('shape color votes', cnt)
    except Exception:
        traceback.print_exc()


A.run_program('re86', prog)

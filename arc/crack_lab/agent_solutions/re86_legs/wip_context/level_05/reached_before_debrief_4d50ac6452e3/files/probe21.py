import json, sys
sys.path.insert(0, '/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
from perception import arr, UP, DOWN, LEFT, RIGHT, USE

PATH = json.load(open('checkpoint.json'))['final_path']


def show(f, r0, r1, c0, c1):
    print('    ' + ''.join(f'{c%10}' for c in range(c0, c1)))
    for r in range(r0, r1):
        print(f'{r:3d} ' + ''.join('.' if f[r, c] == 5 else format(int(f[r, c]), 'x')
                                   for c in range(c0, c1)))


def prog(env):
    for a in PATH:
        env.step(a)
    f = arr(env.frame())
    print('== region rows28-46 cols48-63')
    show(f, 28, 47, 48, 64)
    print('== shape cycle')
    scout = env.clone()
    first = None
    for i in range(6):
        before = arr(scout.frame()).copy()
        blk = list(zip(*((before == 0).nonzero())))
        ctr = tuple(int(v) for v in blk[0]) if len(blk) == 1 else blk
        if first is None:
            first = ctr
        elif ctr == first:
            print('cycle closed after', i)
            break
        mv = scout.clone(); mv.step(RIGHT)
        after = arr(mv.frame())
        vanished = {}
        for r, c in zip(*((before != after).nonzero())):
            if int(after[r, c]) == 5 and int(before[r, c]) not in (0, 5):
                vanished.setdefault(int(before[r, c]), []).append((int(r), int(c)))
        print(f'[{i}] center={ctr} vanished_colors=' +
              str({k: len(v) for k, v in vanished.items()}))
        scout.step(USE)


A.run_program('re86', prog)

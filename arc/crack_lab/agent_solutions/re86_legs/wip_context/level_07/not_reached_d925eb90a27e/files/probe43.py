import json, sys
sys.path.insert(0, '/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
from perception import arr

PATH = json.load(open('checkpoint.json'))['final_path']
UP, DOWN, LEFT, RIGHT, USE = 1, 2, 3, 4, 5
RECT = [UP]*3 + [RIGHT]*4 + [UP]*8 + [RIGHT]*9 + [DOWN]*10
CROSS = ([USE] + [DOWN]*5 + [LEFT]*3 + [DOWN]*6 + [LEFT]*3
         + [UP]*3 + [LEFT]*6 + [UP]*10)
MARK11 = ((30,45), (30,54), (57,45), (57,54))
MARK9 = ((6,12), (9,9), (9,30), (27,12))


def covered(e, marks, color):
    f = arr(e.frame())
    return sum(int(f[p]) == color for p in marks)


def prog(env):
    for action in PATH:
        env.step(action)
    c = env.clone()
    for action in RECT:
        c.step(action)
    print('rect', covered(c, MARK11, 11), 'lvl', c.levels_completed)
    for i, action in enumerate(CROSS):
        c.step(action)
        if i in (0, 5, 8, 14, 17, 20, 26, 36):
            zeros = list(zip(*((arr(c.frame()) == 0).nonzero())))
            print(i, action, 'dot', zeros, 'cov9', covered(c, MARK9, 9),
                  'cov11', covered(c, MARK11, 11), 'lvl', c.levels_completed)
    print('FINAL moves', len(RECT) + len(CROSS), 'lvl', c.levels_completed,
          'terminal', c.terminal())


A.run_program('re86', prog)

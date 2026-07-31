import json, sys, traceback
sys.path.insert(0, '/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
from perception import arr, connected_components

PATH = json.load(open('checkpoint.json'))['final_path']
UP, DOWN, LEFT, RIGHT, USE = 1, 2, 3, 4, 5


def dot(e):
    f = arr(e.frame())
    ys, xs = (f == 0).nonzero()
    return (int(ys[0]), int(xs[0])) if len(ys) else None


def prog(env):
  try:
    for a in PATH:
        env.step(a)
    c = env.clone()
    plan = [UP]*3 + [RIGHT]*11
    for i, a in enumerate(plan):
        try:
            c.step(a)
        except Exception as ex:
            print(i, a, 'STEP RAISED', type(ex).__name__, ex, 'term', c.terminal())
            return
        try:
            print(i, a, 'dot', dot(c), 'lvl', c.levels_completed)
        except Exception as ex:
            print(i, a, 'FRAME RAISED', type(ex).__name__, 'term', c.terminal(),
                  'lvl', c.levels_completed)
            return
  except Exception:
    traceback.print_exc()


A.run_program('re86', prog)

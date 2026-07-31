import json, sys, traceback
sys.path.insert(0, '/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
from perception import arr, connected_components, color_counts

PATH = json.load(open('checkpoint.json'))['final_path']
UP, DOWN, LEFT, RIGHT, USE = 1, 2, 3, 4, 5


def safe(e):
    try:
        return arr(e.frame())
    except Exception as ex:
        return None


def dot(f):
    ys, xs = (f == 0).nonzero()
    return (int(ys[0]), int(xs[0])) if len(ys) else None


def prog(env):
  try:
    for a in PATH:
        env.step(a)
    print('start lvl', env.levels_completed)

    print('\n=== trace: square UP x6 ===')
    c = env.clone()
    for i in range(7):
        f = safe(c)
        if f is None:
            print(i, 'NO FRAME; terminal', c.terminal()); break
        print(i, 'dot', dot(f), 'lvl', c.levels_completed, 'term', c.terminal(),
              'c11', int((f == 11).sum()), 'hud15', int((f[63] == 15).sum()))
        c.step(UP)

    print('\n=== trace: square RIGHT x14 ===')
    c = env.clone()
    for i in range(15):
        f = safe(c)
        if f is None:
            print(i, 'NO FRAME; terminal', c.terminal()); break
        print(i, 'dot', dot(f), 'lvl', c.levels_completed, 'term', c.terminal(),
              'c11', int((f == 11).sum()))
        c.step(RIGHT)
  except Exception:
    traceback.print_exc()


A.run_program('re86', prog)

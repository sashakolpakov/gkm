import json, sys, traceback
sys.path.insert(0, '/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
from perception import arr, connected_components, color_counts

PATH = json.load(open('checkpoint.json'))['final_path']

SEL = 0  # selection dot colour


def sel(e):
    f = arr(e.frame())
    ys, xs = (f == SEL).nonzero()
    return [(int(y), int(x)) for y, x in zip(ys, xs)]


def summary(e):
    f = arr(e.frame())
    return color_counts(f)


def prog(env):
  try:
    for a in PATH:
        env.step(a)
    print('=== USE cycle (selection dot positions) ===')
    c = env.clone()
    for i in range(10):
        print(i, sel(c), 'lvl', c.levels_completed)
        c.step(5)

    print('\n=== per-selection: which action moves what ===')
    c = env.clone()
    for s in range(3):
        base = arr(c.frame())
        print(f'--- selection {s} dot={sel(c)} ---')
        for a in (1, 2, 3, 4):
            d = c.clone(); d.step(a)
            df = arr(d.frame())
            ys, xs = (base != df).nonzero()
            changed = sorted({(int(base[y, x]), int(df[y, x])) for y, x in zip(ys, xs)})
            print('  a', a, 'ndiff', len(ys), 'transitions', changed[:8])
        c.step(5)

    print('\n=== colour counts stability: walk selection0 (square,11) up 4x ===')
    c = env.clone()
    for i in range(5):
        print(i, summary(c), 'dot', sel(c))
        c.step(1)
  except Exception:
    traceback.print_exc()


A.run_program('re86', prog)

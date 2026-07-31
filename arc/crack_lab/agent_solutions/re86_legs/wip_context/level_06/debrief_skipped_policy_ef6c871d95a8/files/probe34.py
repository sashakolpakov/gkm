import json, sys, traceback
sys.path.insert(0, '/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
from perception import arr, connected_components, color_counts

PATH = json.load(open('checkpoint.json'))['final_path']
UP, DOWN, LEFT, RIGHT, USE = 1, 2, 3, 4, 5


def dot(e):
    f = arr(e.frame())
    ys, xs = (f == 0).nonzero()
    return (int(ys[0]), int(xs[0])) if len(ys) else None


def markers(e):
    f = arr(e.frame())
    out = {}
    for b in connected_components(f, colors=(4,), min_area=8):
        if b.size == (3, 3) and b.area == 8:
            r, c = b.bbox[0] + 1, b.bbox[1] + 1
            out[(r, c)] = int(f[r, c])
    return out


def hud(e):
    f = arr(e.frame())
    return int((f[63] == 15).sum())


def show(e, tag):
    print(f'{tag}: dot={dot(e)} lvl={e.levels_completed} hud={hud(e)} '
          f'markers={sorted(markers(e).items())}')


def goto(e, target):
    """Move current selection to target centre on the step-3 lattice."""
    while True:
        d = dot(e)
        if d == target:
            return
        dr, dc = target[0] - d[0], target[1] - d[1]
        if dr:
            e.step(UP if dr < 0 else DOWN)
        elif dc:
            e.step(LEFT if dc < 0 else RIGHT)
        else:
            return


def prog(env):
  try:
    for a in PATH:
        env.step(a)
    show(env, 'start')

    print('\n=== T1: square top edge onto the two row-30 markers (Cr=39,Cc=48) ===')
    c = env.clone()
    goto(c, (39, 48))
    show(c, ' at(39,48)')
    f = arr(c.frame())
    for r in (29, 30, 31):
        print('  row', r, ''.join('.' if int(v) == 5 else format(int(v), 'x')
                                  for v in f[r][40:60]))
    print('\n  -- now move away and re-inspect (permanence test) --')
    goto(c, (48, 15))
    show(c, ' back')

    print('\n=== T2: drive square centre onto the colour-1 fixture ===')
    for tgt in [(30, 30), (33, 33), (30, 33), (33, 30)]:
        d = env.clone()
        goto(d, tgt)
        cc = color_counts(arr(d.frame()))
        print(f'  square->{tgt}: lvl={d.levels_completed} colors={cc}')

    print('\n=== T3: drive plus centre onto the colour-1 fixture ===')
    for tgt in [(30, 30), (33, 33), (30, 33), (33, 30)]:
        d = env.clone(); d.step(USE)
        goto(d, tgt)
        cc = color_counts(arr(d.frame()))
        print(f'  plus->{tgt}: lvl={d.levels_completed} colors={cc}')

    print('\n=== T4: HUD depletion rate ===')
    c = env.clone()
    for i in range(0, 13):
        if i:
            c.step(UP if i % 2 else DOWN)
        print('  moves', i, 'hud', hud(c))
  except Exception:
    traceback.print_exc()


A.run_program('re86', prog)

import json, sys, traceback
sys.path.insert(0, '/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
from perception import arr, connected_components

PATH = json.load(open('checkpoint.json'))['final_path']
UP, DOWN, LEFT, RIGHT, USE = 1, 2, 3, 4, 5
NAME = {1: 'U', 2: 'D', 3: 'L', 4: 'R', 5: 'S'}


def shape(e, color):
    f = arr(e.frame())
    bs = [b for b in connected_components(f, colors=(color,), min_area=10)]
    if not bs:
        return None
    b = max(bs, key=lambda x: x.area)
    r0, c0, r1, c1 = b.bbox
    return (r0, c0, r1, c1, r1 - r0 + 1, c1 - c0 + 1, b.area)


def run(env, plan, color, tag):
    c = env.clone()
    print(f'--- {tag} ---')
    s = shape(c, color)
    print(f'  init  bbox={s[:4]} h={s[4]} w={s[5]} h+w={s[4]+s[5]} area={s[6]}')
    for a in plan:
        try:
            c.step(a)
            s = shape(c, color)
        except Exception as ex:
            print('  DIED', type(ex).__name__); return
        if s is None:
            print(f'  {NAME[a]} -> shape gone'); continue
        print(f'  {NAME[a]} bbox={s[:4]} h={s[4]} w={s[5]} h+w={s[4]+s[5]} area={s[6]}')


def prog(env):
  try:
    for a in PATH:
        env.step(a)
    run(env, [DOWN]*4, 11, 'square DOWN x4 (toward board bottom)')
    run(env, [LEFT]*5, 11, 'square LEFT x5 (toward left edge)')
    run(env, [UP]*5, 11, 'square UP x5 (open)')
    c = env.clone(); c.step(USE)
    run(c, [UP]*3, 9, 'plus UP x3 (toward top edge)')
    run(c, [RIGHT]*3, 9, 'plus RIGHT x3 (toward right edge)')
    run(c, [LEFT]*4, 9, 'plus LEFT x4 (open)')
  except Exception:
    traceback.print_exc()


A.run_program('re86', prog)

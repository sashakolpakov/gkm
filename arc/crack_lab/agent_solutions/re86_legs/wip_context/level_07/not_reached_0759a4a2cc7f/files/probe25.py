import json, sys
sys.path.insert(0, '/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
from perception import arr, USE, LEFT, RIGHT, UP, DOWN

PATH = json.load(open('checkpoint.json'))['final_path']

PLAN = [((42, 24), (54, 6), (15, 30)),
        ((18, 30), (54, 57), (36, 51)),
        ((33, 54), (54, 6), (51, 33))]


def ctr(e):
    f = arr(e.frame())
    b = list(zip(*((f == 0).nonzero())))
    return tuple(int(v) for v in b[0]) if len(b) == 1 else None


def goto(e, dst, tag):
    cur = ctr(e)
    for _ in range(200):
        if cur == dst:
            return True
        a = (DOWN if dst[0] > cur[0] else UP) if cur[0] != dst[0] else (
            RIGHT if dst[1] > cur[1] else LEFT)
        e.step(a)
        new = ctr(e)
        if new == cur:
            print(f'  BLOCKED {tag} at {cur} -> {dst} action {a}')
            return False
        cur = new
    return False


def prog(env):
  import traceback
  try:
      for a in PATH:
          env.step(a)
      s = env.clone()
      for i, (start, paint, dst) in enumerate(PLAN):
          if i:
              s.step(USE)
          print(f'shape{i} center={ctr(s)} expect {start}')
          goto(s, paint, f's{i}-paint')
          print('   at paint', ctr(s), 'lvl', s.levels_completed)
          goto(s, dst, f's{i}-dst')
          print('   at dst', ctr(s), 'lvl', s.levels_completed)
      print('FINAL levels', s.levels_completed, 'terminal', s.terminal())


  except Exception:
    traceback.print_exc()
A.run_program('re86', prog)

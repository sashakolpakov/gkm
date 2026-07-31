import json, sys, traceback
sys.path.insert(0, '/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
from perception import arr, connected_components, color_counts, frame_delta

PATH = json.load(open('checkpoint.json'))['final_path']


def prog(env):
  try:
    for a in PATH:
        env.step(a)
    print('levels', env.levels_completed, 'terminal', env.terminal(),
          'actions', getattr(env, 'actions', None))
    f = arr(env.frame())
    print('shape', f.shape, 'colors', color_counts(f))
    print('--- blobs (area<400) ---')
    for b in connected_components(f, min_area=1):
        if b.area < 400:
            print(f"c={b.color:2d} bbox={b.bbox} sz={b.size} area={b.area}")
    print('--- action deltas ---')
    for a in (1, 2, 3, 4, 5):
        c = env.clone(); c.step(a)
        d = frame_delta(f, c.frame())
        print(a, 'count', d['count'], 'bbox', d['bbox'], 'lvl', c.levels_completed)
    print('--- row occupancy map (non-background) ---')
    from collections import Counter
    bg = Counter(int(v) for v in f.ravel()).most_common(1)[0][0]
    print('bg', bg)
    for r in range(f.shape[0]):
        row = ''.join('.' if int(v) == bg else format(int(v), 'x') for v in f[r])
        print(f'{r:2d} {row}')
  except Exception:
    traceback.print_exc()


A.run_program('re86', prog)

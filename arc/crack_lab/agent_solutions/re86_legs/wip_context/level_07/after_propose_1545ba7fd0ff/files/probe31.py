import json, sys, traceback
sys.path.insert(0, '/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import legs as L
from perception import arr, connected_components

PATH = json.load(open('checkpoint.json'))['final_path']


def prog(env):
  try:
    for a in PATH:
        env.step(a)
    frame = arr(env.frame())
    rows, cols = frame.shape[:2]
    rings = [b for b in connected_components(frame, colors=(4,), min_area=8)
             if b.size == (3, 3) and b.area == 8]
    step = rings[0].size[0]
    starts = L._selection_cycle(env)
    dist = max(rows, cols) // step + 1
    print('starts', starts, 'step', step, 'dist', dist)
    markers, swatches = L._survey_markers_and_swatches(env, len(starts), dist, 4, 2)
    print('markers', len(markers), sorted(markers.items()))
    print('swatches', {k: (len(v), min(v), max(v)) for k, v in swatches.items()})
    for i in range(len(starts)):
        ctr, sc, offs = L._shape_offsets(env, i, len(starts), dist)
        print(f'shape{i} ctr={ctr} n_off={len(offs)} color={sc} '
              f'dr=[{min(o[0] for o in offs)},{max(o[0] for o in offs)}] '
              f'dc=[{min(o[1] for o in offs)},{max(o[1] for o in offs)}]')
        prev = L._paint_routes(offs, ctr, sc, swatches, step, rows-1, cols)
        groups = {}
        for (pt, col) in prev:
            cov = frozenset(m for m in markers
                            if (m[0]-pt[0], m[1]-pt[1]) in offs)
            if not cov:
                continue
            if {markers[m] for m in cov} != {col}:
                continue
            groups.setdefault(cov, []).append((pt, col))
        print('   valid groups', len(groups))
        for g, v in sorted(groups.items(), key=lambda kv: -len(kv[0]))[:6]:
            print('     ', sorted(g), 'e.g.', v[0])
  except Exception:
    traceback.print_exc()


A.run_program('re86', prog)

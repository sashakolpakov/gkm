import json, sys, traceback
sys.path.insert(0, '/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import numpy as np
from perception import arr, connected_components
import legs

with open("checkpoint.json") as f:
    PATH = json.load(f)["final_path"]

STORE = {}

def solve(env):
    for a in PATH:
        env.step(a)
    STORE['lvl0'] = env.levels_completed
    # Inspect structure programmatically
    f = arr(env.frame())
    from collections import Counter, defaultdict
    bg = Counter(int(v) for v in f.flat).most_common(1)[0][0]
    STORE['bg'] = bg
    rings = [b for b in connected_components(f, colors=(4,), min_area=8)
             if b.size == (3,3) and b.area == 8]
    STORE['nrings'] = len(rings)
    tgt = defaultdict(list)
    for ring in rings:
        p = (ring.bbox[0]+1, ring.bbox[1]+1)
        tgt[int(f[p])].append(p)
    STORE['targets'] = {k: v for k, v in tgt.items()}
    # stations
    stations = {}
    for border in connected_components(f, colors=(2,), min_area=20):
        if border.size != (6,6) or border.area != 20:
            continue
        r0,c0,r1,c1 = border.bbox
        interior = f[r0+1:r1, c0+1:c1]
        cols = {int(v) for v in interior.flat if int(v) not in (bg, 2)}
        if len(cols)==1:
            stations[cols.pop()] = border.bbox
    STORE['stations'] = stations
    # find black-pixel selected shape and cycle via USE
    scout = env.clone()
    centers = []
    for _ in range(10):
        bf = arr(scout.frame())
        bp = list(zip(*((bf==0).nonzero())))
        if len(bp)!=1:
            centers.append(('noblack', len(bp)))
            break
        c = tuple(int(v) for v in bp[0])
        if centers and c == centers[0]:
            break
        centers.append(c)
        scout.step(5)
    STORE['shape_centers'] = centers
    raise SystemExit

try:
    A.run_program('re86', solve)
except SystemExit:
    pass
except Exception:
    traceback.print_exc()

for k,v in STORE.items():
    print(k, "=", v)

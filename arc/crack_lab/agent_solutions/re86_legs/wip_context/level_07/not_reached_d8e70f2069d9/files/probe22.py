import json, sys
sys.path.insert(0, '/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
from perception import arr, connected_components, UP, DOWN, LEFT, RIGHT, USE

PATH = json.load(open('checkpoint.json'))['final_path']


def prog(env):
    for a in PATH:
        env.step(a)
    scout = env.clone()
    # park every shape in a corner-ish spot far from the play area
    for i in range(3):
        if i:
            scout.step(USE)
        for _ in range(7):
            scout.step(UP)
        for _ in range(7):
            scout.step(LEFT)
    f = arr(scout.frame())
    rings = [b for b in connected_components(f, colors=(4,), min_area=8)
             if b.size == (3, 3) and b.area == 8]
    print('rings', len(rings))
    from collections import defaultdict
    g = defaultdict(list)
    for b in rings:
        p = (b.bbox[0] + 1, b.bbox[1] + 1)
        g[int(f[p])].append(p)
    for k in sorted(g):
        print('marker color', k, sorted(g[k]))
    other4 = [b for b in connected_components(f, colors=(4,), min_area=1)
              if not (b.size == (3, 3) and b.area == 8)]
    print('non-ring 4 blobs', [(b.bbox, b.area) for b in other4])


A.run_program('re86', prog)
